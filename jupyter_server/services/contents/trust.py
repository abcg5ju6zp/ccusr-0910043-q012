"""笔记本签名信任的扩展管理。

在 nbformat 的 ``NotebookNotary`` 之上提供面向密钥轮换的信任管理：

- 签名记录携带密钥代际、签发时间与内容规范化版本；
- 轮换密钥时可设置验证窗口（verify window）与撤销时点（revocation point），
  用于区分签名产生于密钥泄露之前还是之后；
- 另存、恢复检查点、外部修改与跨目录复制之后重新判断信任继承；
- 验证过程只对内容做规范化哈希，绝不执行笔记本的代码或输出内容；
- 重签任务可中断后继续，且不会把已经撤销的签名恢复为可信。
"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import typing as t
import uuid
from base64 import b64decode, b64encode
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hmac import HMAC
from pathlib import Path

import nbformat
from jupyter_core.application import JupyterApp
from traitlets import Callable, Enum, Integer, Unicode, default
from traitlets.config import LoggingConfigurable

#: 当前的内容规范化版本。升级规范化方案时递增，
#: 旧版本签名的记录会被判定为 "stale"，需要重签。
NORMALIZATION_VERSION = 1

#: 仍然受支持（可以重算摘要）的规范化版本集合
SUPPORTED_NORMALIZATION_VERSIONS = frozenset({1})

#: 允许用于 HMAC 的哈希算法（shake 系列需要长度参数，不兼容 HMAC）
ALGORITHMS = [a for a in hashlib.algorithms_guaranteed if not a.startswith("shake_")]


class TrustState:
    """信任判定结果。"""

    TRUSTED = "trusted"  # 签名有效，且在验证窗口与撤销时点约束之内
    UNKNOWN = "unknown"  # 没有找到与内容匹配的签名记录
    REVOKED = "revoked"  # 签名已被撤销，或产生于撤销时点之后，或密钥代际被整体撤销
    EXPIRED = "expired"  # 旧密钥代际的验证窗口已关闭
    STALE = "stale"  # 内容规范化版本过旧，需要重签


#: 判定优先级：撤销永远优先于可信，保证已撤销的签名不会被恢复为可信
_STATE_PRIORITY = {
    TrustState.REVOKED: 0,
    TrustState.TRUSTED: 1,
    TrustState.EXPIRED: 2,
    TrustState.STALE: 3,
    TrustState.UNKNOWN: 4,
}


def _utcnow() -> datetime:
    """返回带时区的当前 UTC 时间。"""
    return datetime.now(tz=timezone.utc)


def _coerce_datetime(value: datetime | str) -> datetime:
    """把 ISO 字符串或 naive datetime 规范化为带 UTC 时区的 datetime。"""
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def _yield_normalized(obj: t.Any) -> t.Iterator[bytes]:
    """把可 JSON 序列化的对象展开为字节流。

    与 nbformat.sign.yield_everything 相同的规范化方案（v1）：
    字典按键排序，字符串按 UTF-8 编码，其余值取其 ``str()``。
    """
    if isinstance(obj, dict):
        for key in sorted(obj):
            value = obj[key]
            assert isinstance(key, str)
            yield key.encode()
            yield from _yield_normalized(value)
    elif isinstance(obj, (list, tuple)):
        for element in obj:
            yield from _yield_normalized(element)
    elif isinstance(obj, str):
        yield obj.encode("utf8")
    else:
        yield str(obj).encode("utf8")


def _strip_runtime_flags(node: t.Any) -> t.Any:
    """移除运行时信任标记（cell.metadata.trusted），返回新的对象副本。

    trusted 标记由信任判定流程写入/弹出，不属于签名内容；把它纳入哈希
    会导致同一份内容在内存中与磁盘上得到不同的摘要。
    """
    if isinstance(node, dict):
        result = {key: _strip_runtime_flags(value) for key, value in node.items()}
        if "cell_type" in result and isinstance(result.get("metadata"), dict):
            metadata = dict(result["metadata"])
            metadata.pop("trusted", None)
            result["metadata"] = metadata
        return result
    if isinstance(node, (list, tuple)):
        return [_strip_runtime_flags(item) for item in node]
    return node


def normalize_notebook(nb: t.Any, version: int = NORMALIZATION_VERSION) -> t.Iterator[bytes]:
    """按指定的内容规范化版本，把笔记本展开为用于哈希的字节流。

    只读取笔记本的静态内容（源码、输出、元数据），绝不执行任何单元格
    代码或输出内容；旧的签名元数据与运行时 trusted 标记不参与哈希。
    不会修改传入的笔记本。
    """
    if version not in SUPPORTED_NORMALIZATION_VERSIONS:
        msg = f"不支持的内容规范化版本: {version}"
        raise ValueError(msg)
    view = _strip_runtime_flags(dict(nb))
    metadata = dict(view.get("metadata") or {})
    metadata.pop("signature", None)
    view["metadata"] = metadata
    yield from _yield_normalized(view)


@dataclass
class KeyGeneration:
    """一代签名密钥的元数据。"""

    generation: int
    created_at: datetime
    status: str = "active"  # active | rotated | revoked
    verify_until: datetime | None = None  # 验证窗口终点（仅轮换后的旧代际使用）
    revoked_before: datetime | None = None  # 撤销时点：该时点及之后的签名视为泄露后伪造


@dataclass
class SignatureRecord:
    """一条签名记录：内容哈希、HMAC 摘要与密钥代际等上下文。"""

    content_hash: str
    digest: str
    algorithm: str
    key_generation: int
    issued_at: datetime
    normalization_version: int
    path: str = ""
    status: str = "active"  # active | revoked
    revoked_at: datetime | None = None


@dataclass
class TrustVerdict:
    """一次信任判定的结果。"""

    state: str
    trusted: bool
    reason: str
    record: SignatureRecord | None = None
    event: str = "read"
    external_change: bool = False  # 是否检测到管理器之外的内容修改


class ResignInterrupted(Exception):
    """重签任务被中断；进度已持久化，可稍后继续。"""

    def __init__(self, task_id: str):
        super().__init__(f"重签任务 {task_id} 已中断，可通过 resume_resign 继续")
        self.task_id = task_id


class TrustStore:
    """签名记录与重签进度的持久化后端接口。"""

    def store_record(self, record: SignatureRecord) -> None:
        """保存一条签名记录；已撤销的记录不得被恢复为 active。"""
        raise NotImplementedError

    def records_for_hash(self, content_hash: str) -> list[SignatureRecord]:
        """返回某一内容哈希对应的全部签名记录。"""
        raise NotImplementedError

    def revoke_hash(self, content_hash: str, revoked_at: datetime) -> int:
        """撤销某一内容哈希的全部签名记录，返回撤销条数。"""
        raise NotImplementedError

    def get_path_state(self, path: str) -> str | None:
        """返回某路径最近一次见到的内容哈希，用于检测外部修改。"""
        raise NotImplementedError

    def set_path_state(self, path: str, content_hash: str) -> None:
        """记录某路径最近一次见到的内容哈希。"""
        raise NotImplementedError

    def create_resign_task(self, task_id: str, paths: list[str], created_at: datetime) -> None:
        """登记一个重签任务及其待处理路径（幂等）。"""
        raise NotImplementedError

    def resign_items(self, task_id: str, status: str | None = None) -> list[dict[str, t.Any]]:
        """列出重签任务的条目，可按状态过滤。"""
        raise NotImplementedError

    def update_resign_item(self, task_id: str, path: str, status: str, detail: str = "") -> None:
        """更新重签条目的状态。"""
        raise NotImplementedError

    def resign_task_ids(self) -> list[str]:
        """列出全部重签任务 ID。"""
        raise NotImplementedError

    def close(self) -> None:
        """关闭后端持有的资源。"""


class MemoryTrustStore(TrustStore):
    """内存版信任存储（无 SQLite 时的回退，进程退出即丢失）。"""

    def __init__(self) -> None:
        self.records: dict[tuple[str, str], SignatureRecord] = {}
        self.path_states: dict[str, str] = {}
        self.resign: dict[str, dict[str, dict[str, str]]] = {}

    def store_record(self, record: SignatureRecord) -> None:
        key = (record.digest, record.algorithm)
        existing = self.records.get(key)
        if existing is not None and existing.status == "revoked":
            # 已撤销的签名不得恢复为可信
            return
        self.records[key] = record

    def records_for_hash(self, content_hash: str) -> list[SignatureRecord]:
        return [r for r in self.records.values() if r.content_hash == content_hash]

    def revoke_hash(self, content_hash: str, revoked_at: datetime) -> int:
        count = 0
        for record in self.records.values():
            if record.content_hash == content_hash and record.status != "revoked":
                record.status = "revoked"
                record.revoked_at = revoked_at
                count += 1
        return count

    def get_path_state(self, path: str) -> str | None:
        return self.path_states.get(path)

    def set_path_state(self, path: str, content_hash: str) -> None:
        self.path_states[path] = content_hash

    def create_resign_task(self, task_id: str, paths: list[str], created_at: datetime) -> None:
        items = self.resign.setdefault(task_id, {})
        for path in paths:
            items.setdefault(
                path,
                {"status": "pending", "detail": "", "updated_at": created_at.isoformat()},
            )

    def resign_items(self, task_id: str, status: str | None = None) -> list[dict[str, t.Any]]:
        items = self.resign.get(task_id, {})
        result = []
        for path, item in items.items():
            if status is not None and item["status"] != status:
                continue
            result.append({"path": path, **item})
        return result

    def update_resign_item(self, task_id: str, path: str, status: str, detail: str = "") -> None:
        item = self.resign.get(task_id, {}).get(path)
        if item is not None:
            item["status"] = status
            item["detail"] = detail
            item["updated_at"] = _utcnow().isoformat()

    def resign_task_ids(self) -> list[str]:
        return list(self.resign)


class SQLiteTrustStore(TrustStore, LoggingConfigurable):
    """SQLite 版信任存储。"""

    def __init__(self, db_file: str, **kwargs: t.Any):
        super().__init__(**kwargs)
        self.db_file = db_file
        self.db = self._connect_db(db_file)

    def close(self) -> None:
        if self.db is not None:
            self.db.close()

    def _connect_db(self, db_file: str) -> sqlite3.Connection:
        db = None
        try:
            db = sqlite3.connect(db_file)
            self.init_db(db)
        except (sqlite3.DatabaseError, sqlite3.OperationalError):
            if db_file == ":memory:":
                raise
            old_db_location = db_file + ".bak"
            if db is not None:
                db.close()
            self.log.warning(
                "信任签名库 %s 无法打开（可能已损坏），已改名为 %s 并重建。",
                db_file,
                old_db_location,
            )
            try:
                Path(db_file).rename(old_db_location)
                db = sqlite3.connect(db_file)
                self.init_db(db)
            except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError):
                if db is not None:
                    db.close()
                self.log.warning("信任签名库无法写入磁盘，本会话改用内存库。")
                self.db_file = ":memory:"
                db = sqlite3.connect(":memory:")
                self.init_db(db)
        return db

    def init_db(self, db: sqlite3.Connection) -> None:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS trust_signatures
            (
                id integer PRIMARY KEY AUTOINCREMENT,
                content_hash text NOT NULL,
                digest text NOT NULL,
                algorithm text NOT NULL,
                key_generation integer NOT NULL,
                issued_at text NOT NULL,
                normalization_version integer NOT NULL,
                path text,
                status text NOT NULL DEFAULT 'active',
                revoked_at text,
                last_seen text,
                UNIQUE (digest, algorithm)
            )"""
        )
        db.execute("CREATE INDEX IF NOT EXISTS trust_sig_hash ON trust_signatures(content_hash)")
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS trust_path_state
            (
                path text PRIMARY KEY,
                content_hash text NOT NULL,
                updated_at text NOT NULL
            )"""
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS resign_items
            (
                task_id text NOT NULL,
                path text NOT NULL,
                status text NOT NULL DEFAULT 'pending',
                detail text,
                updated_at text,
                PRIMARY KEY (task_id, path)
            )"""
        )
        db.commit()

    @staticmethod
    def _row_to_record(row: tuple) -> SignatureRecord:
        (
            _id,
            content_hash,
            digest,
            algorithm,
            key_generation,
            issued_at,
            normalization_version,
            path,
            status,
            revoked_at,
            _last_seen,
        ) = row
        return SignatureRecord(
            content_hash=content_hash,
            digest=digest,
            algorithm=algorithm,
            key_generation=key_generation,
            issued_at=_coerce_datetime(issued_at),
            normalization_version=normalization_version,
            path=path or "",
            status=status,
            revoked_at=_coerce_datetime(revoked_at) if revoked_at else None,
        )

    def store_record(self, record: SignatureRecord) -> None:
        if self.db is None:
            return
        now = _utcnow().isoformat()
        row = self.db.execute(
            "SELECT status FROM trust_signatures WHERE digest = ? AND algorithm = ?",
            (record.digest, record.algorithm),
        ).fetchone()
        if row is not None:
            if row[0] == "revoked":
                # 已撤销的签名不得恢复为可信
                return
            self.db.execute(
                "UPDATE trust_signatures SET path = ?, last_seen = ? "
                "WHERE digest = ? AND algorithm = ?",
                (record.path, now, record.digest, record.algorithm),
            )
        else:
            self.db.execute(
                """
                INSERT INTO trust_signatures
                (content_hash, digest, algorithm, key_generation, issued_at,
                 normalization_version, path, status, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.content_hash,
                    record.digest,
                    record.algorithm,
                    record.key_generation,
                    record.issued_at.isoformat(),
                    record.normalization_version,
                    record.path,
                    record.status,
                    now,
                ),
            )
        self.db.commit()

    def records_for_hash(self, content_hash: str) -> list[SignatureRecord]:
        if self.db is None:
            return []
        rows = self.db.execute(
            "SELECT * FROM trust_signatures WHERE content_hash = ?",
            (content_hash,),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def revoke_hash(self, content_hash: str, revoked_at: datetime) -> int:
        if self.db is None:
            return 0
        cursor = self.db.execute(
            "UPDATE trust_signatures SET status = 'revoked', revoked_at = ? "
            "WHERE content_hash = ? AND status != 'revoked'",
            (revoked_at.isoformat(), content_hash),
        )
        self.db.commit()
        return cursor.rowcount

    def get_path_state(self, path: str) -> str | None:
        if self.db is None:
            return None
        row = self.db.execute(
            "SELECT content_hash FROM trust_path_state WHERE path = ?",
            (path,),
        ).fetchone()
        return row[0] if row else None

    def set_path_state(self, path: str, content_hash: str) -> None:
        if self.db is None:
            return
        self.db.execute(
            "INSERT INTO trust_path_state (path, content_hash, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(path) DO UPDATE SET content_hash = excluded.content_hash, "
            "updated_at = excluded.updated_at",
            (path, content_hash, _utcnow().isoformat()),
        )
        self.db.commit()

    def create_resign_task(self, task_id: str, paths: list[str], created_at: datetime) -> None:
        if self.db is None:
            return
        self.db.executemany(
            "INSERT OR IGNORE INTO resign_items (task_id, path, status, detail, updated_at) "
            "VALUES (?, ?, 'pending', '', ?)",
            [(task_id, path, created_at.isoformat()) for path in paths],
        )
        self.db.commit()

    def resign_items(self, task_id: str, status: str | None = None) -> list[dict[str, t.Any]]:
        if self.db is None:
            return []
        if status is None:
            rows = self.db.execute(
                "SELECT path, status, detail, updated_at FROM resign_items WHERE task_id = ? "
                "ORDER BY rowid",
                (task_id,),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT path, status, detail, updated_at FROM resign_items "
                "WHERE task_id = ? AND status = ? ORDER BY rowid",
                (task_id, status),
            ).fetchall()
        return [
            {"path": path, "status": st, "detail": detail or "", "updated_at": updated_at}
            for path, st, detail, updated_at in rows
        ]

    def update_resign_item(self, task_id: str, path: str, status: str, detail: str = "") -> None:
        if self.db is None:
            return
        self.db.execute(
            "UPDATE resign_items SET status = ?, detail = ?, updated_at = ? "
            "WHERE task_id = ? AND path = ?",
            (status, detail, _utcnow().isoformat(), task_id, path),
        )
        self.db.commit()

    def resign_task_ids(self) -> list[str]:
        if self.db is None:
            return []
        rows = self.db.execute("SELECT DISTINCT task_id FROM resign_items").fetchall()
        return [row[0] for row in rows]


class TrustManager(LoggingConfigurable):
    """笔记本签名信任管理器。

    负责签名记录的签发与验证、密钥代际轮换（验证窗口与撤销时点）、
    信任继承的重新判断，以及可中断续跑的重签任务。
    """

    data_dir = Unicode(help="""信任签名库与密钥环的存储目录。""").tag(config=True)

    @default("data_dir")
    def _data_dir_default(self) -> str:
        app = None
        try:
            if JupyterApp.initialized():
                app = JupyterApp.instance()
        except Exception:  # noqa: BLE001 - 与 nbformat 的默认实现保持一致
            pass
        if app is None:
            app = JupyterApp()
            app.initialize(argv=[])
        return app.data_dir

    db_file = Unicode(help="""信任签名库的 SQLite 文件路径，':memory:' 表示不落盘。""").tag(
        config=True
    )

    @default("db_file")
    def _db_file_default(self) -> str:
        if not self.data_dir:
            return ":memory:"
        return str(Path(self.data_dir) / "nbtrust.db")

    keyring_file = Unicode(help="""保存各代际密钥与轮换策略的密钥环文件。""").tag(config=True)

    @default("keyring_file")
    def _keyring_file_default(self) -> str:
        if not self.data_dir:
            return ""
        return str(Path(self.data_dir) / "notebook_trust_keyring.json")

    algorithm = Enum(
        ALGORITHMS,
        default_value="sha256",
        help="""计算签名所用的哈希算法。""",
    ).tag(config=True)

    normalization_version = Integer(
        NORMALIZATION_VERSION,
        help="""当前使用的内容规范化版本；低于该版本的签名记录需要重签。""",
    ).tag(config=True)

    store_factory = Callable(help="""返回信任存储后端的可调用对象，默认使用 SQLite。""").tag(
        config=True
    )

    @default("store_factory")
    def _store_factory_default(self) -> t.Callable[[], TrustStore]:
        def factory() -> TrustStore:
            return SQLiteTrustStore(self.db_file, parent=self)

        return factory

    def __init__(self, **kwargs: t.Any):
        super().__init__(**kwargs)
        self.store: TrustStore = self.store_factory()
        self._secrets: dict[int, bytes] = {}
        self._generations: dict[int, KeyGeneration] = {}
        self._current = 0
        self._load_keyring()

    # ------------------------------------------------------------------
    # 密钥环：各代际的密钥与轮换策略
    # ------------------------------------------------------------------

    def _load_keyring(self) -> None:
        """加载密钥环；不存在时创建第 1 代密钥。"""
        path = self.keyring_file
        if path and Path(path).exists():
            with Path(path).open(encoding="utf-8") as f:
                payload = json.load(f)
            self._current = int(payload["current_generation"])
            for gen_str, entry in payload["generations"].items():
                gen = int(gen_str)
                self._secrets[gen] = b64decode(entry["secret_b64"])
                self._generations[gen] = KeyGeneration(
                    generation=gen,
                    created_at=_coerce_datetime(entry["created_at"]),
                    status=entry.get("status", "active"),
                    verify_until=(
                        _coerce_datetime(entry["verify_until"])
                        if entry.get("verify_until")
                        else None
                    ),
                    revoked_before=(
                        _coerce_datetime(entry["revoked_before"])
                        if entry.get("revoked_before")
                        else None
                    ),
                )
        else:
            self._current = 1
            self._secrets[1] = os.urandom(1024)
            self._generations[1] = KeyGeneration(generation=1, created_at=_utcnow())
            self._write_keyring()

    def _write_keyring(self) -> None:
        """把密钥环写回磁盘（权限 0o600）。"""
        path = self.keyring_file
        if not path:
            return
        payload = {
            "current_generation": self._current,
            "generations": {
                str(gen): {
                    "secret_b64": b64encode(self._secrets[gen]).decode("ascii"),
                    "created_at": info.created_at.isoformat(),
                    "status": info.status,
                    "verify_until": info.verify_until.isoformat() if info.verify_until else None,
                    "revoked_before": (
                        info.revoked_before.isoformat() if info.revoked_before else None
                    ),
                }
                for gen, info in sorted(self._generations.items())
            },
        }
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
        try:
            Path(path).chmod(0o600)
        except OSError:
            self.log.warning("无法设置密钥环文件权限: %s", path)

    @property
    def current_generation(self) -> int:
        """当前签发所用的密钥代际。"""
        return self._current

    def generation_info(self, generation: int | None = None) -> KeyGeneration:
        """返回指定（默认当前）密钥代际的元数据。"""
        gen = self._current if generation is None else generation
        return self._generations[gen]

    def rotate_key(
        self,
        verify_window: timedelta | float | None = None,
        revoked_before: datetime | str | None = None,
    ) -> int:
        """轮换签名密钥，返回新的密钥代际。

        - ``verify_window``：旧代际签名的验证窗口（timedelta 或秒数），
          窗口关闭后旧代际签名一律不再可信；缺省表示不设置窗口终点。
        - ``revoked_before``：旧代际的撤销时点，该时点及之后签发的签名
          视为密钥泄露后的伪造，一律不可信；之前的签名在窗口内仍可信。
        """
        now = _utcnow()
        old = self._generations[self._current]
        old.status = "rotated"
        if verify_window is not None:
            if isinstance(verify_window, (int, float)):
                verify_window = timedelta(seconds=verify_window)
            old.verify_until = now + verify_window
        if revoked_before is not None:
            old.revoked_before = _coerce_datetime(revoked_before)
        new_generation = self._current + 1
        self._secrets[new_generation] = os.urandom(1024)
        self._generations[new_generation] = KeyGeneration(generation=new_generation, created_at=now)
        self._current = new_generation
        self._write_keyring()
        self.log.warning(
            "签名密钥已轮换到第 %d 代；第 %d 代验证窗口截止 %s，撤销时点 %s",
            new_generation,
            old.generation,
            old.verify_until.isoformat() if old.verify_until else "<未设置>",
            old.revoked_before.isoformat() if old.revoked_before else "<未设置>",
        )
        return new_generation

    def revoke_generation(self, generation: int) -> None:
        """整体撤销一个密钥代际：该代际的全部签名立即不可信。"""
        info = self._generations[generation]
        info.status = "revoked"
        self._write_keyring()
        self.log.warning("密钥代际 %d 已被整体撤销", generation)

    # ------------------------------------------------------------------
    # 规范化哈希与签名
    # ------------------------------------------------------------------

    def compute_content_hash(self, nb: t.Any, version: int | None = None) -> str:
        """计算规范化内容的（无密钥）SHA-256 哈希，用于记录索引与变更检测。"""
        version = self.normalization_version if version is None else version
        h = hashlib.sha256()
        for b in normalize_notebook(nb, version):
            h.update(b)
        return h.hexdigest()

    def _content_hashes(self, nb: t.Any) -> dict[int, str]:
        """按全部受支持的规范化版本计算内容哈希。

        查找与撤销签名时必须覆盖所有版本，避免规范化方案升级后
        已撤销的签名被绕过恢复为可信。
        """
        versions = {v for v in SUPPORTED_NORMALIZATION_VERSIONS if v <= self.normalization_version}
        versions.add(self.normalization_version)
        return {v: self.compute_content_hash(nb, v) for v in sorted(versions)}

    def compute_digest(self, nb: t.Any, generation: int | None = None) -> str:
        """用指定（默认当前）代际的密钥计算规范化内容的 HMAC 摘要。"""
        gen = self._current if generation is None else generation
        secret = self._secrets[gen]
        hmac = HMAC(secret, digestmod=getattr(hashlib, self.algorithm))
        for b in normalize_notebook(nb, self.normalization_version):
            hmac.update(b)
        return hmac.hexdigest()

    def sign_notebook(self, nb: t.Any, path: str = "") -> SignatureRecord | None:
        """用当前密钥代际为笔记本签发签名记录。

        如果内容命中已撤销的签名，拒绝签发并返回 None——
        已经撤销的签名不得恢复为可信。
        """
        content_hash = self.compute_content_hash(nb)
        for h in self._content_hashes(nb).values():
            for record in self.store.records_for_hash(h):
                if record.status == "revoked":
                    self.log.warning(
                        "内容 %s 的签名已被撤销，拒绝重新签发（路径 %s）",
                        content_hash[:12],
                        path,
                    )
                    return None
        record = SignatureRecord(
            content_hash=content_hash,
            digest=self.compute_digest(nb),
            algorithm=self.algorithm,
            key_generation=self._current,
            issued_at=_utcnow(),
            normalization_version=self.normalization_version,
            path=path,
        )
        self.store.store_record(record)
        if path:
            self.store.set_path_state(path, content_hash)
        return record

    def track_path_state(self, nb: t.Any, path: str = "") -> None:
        """记录路径当前的内容哈希。

        保存流程调用，避免把管理器自身的写入误判为外部修改。
        """
        if path:
            self.store.set_path_state(path, self.compute_content_hash(nb))

    def revoke_notebook(
        self,
        nb: t.Any = None,
        *,
        content_hash: str | None = None,
        path: str = "",
    ) -> int:
        """撤销某一内容对应的全部签名记录（覆盖所有规范化版本），返回撤销条数。"""
        if content_hash is None:
            if nb is None:
                msg = "revoke_notebook 需要 nb 或 content_hash 之一"
                raise ValueError(msg)
            hashes = set(self._content_hashes(nb).values())
        else:
            hashes = {content_hash}
        count = 0
        for h in hashes:
            count += self.store.revoke_hash(h, _utcnow())
        self.log.warning(
            "已撤销内容 %s 的 %d 条签名记录（路径 %s）",
            (content_hash or next(iter(hashes)))[:12],
            count,
            path,
        )
        return count

    # ------------------------------------------------------------------
    # 验证与信任继承重判
    # ------------------------------------------------------------------

    def _evaluate_record(self, record: SignatureRecord, now: datetime) -> tuple[str, str]:
        """按轮换策略评估一条签名记录，返回 (状态, 原因)。"""
        if record.status == "revoked":
            return TrustState.REVOKED, "签名已被撤销"
        gen = self._generations.get(record.key_generation)
        if gen is None:
            return TrustState.REVOKED, f"密钥代际 {record.key_generation} 已不可用"
        if gen.status == "revoked":
            return TrustState.REVOKED, f"密钥代际 {record.key_generation} 已被整体撤销"
        if gen.revoked_before is not None and record.issued_at >= gen.revoked_before:
            return (
                TrustState.REVOKED,
                (
                    f"签名签发时间 {record.issued_at.isoformat()} 不早于撤销时点 "
                    f"{gen.revoked_before.isoformat()}，可能由已泄露密钥伪造"
                ),
            )
        if (
            gen.generation != self._current
            and gen.verify_until is not None
            and now > gen.verify_until
        ):
            return (
                TrustState.EXPIRED,
                (
                    f"密钥代际 {record.key_generation} 的验证窗口已于 "
                    f"{gen.verify_until.isoformat()} 关闭"
                ),
            )
        if record.normalization_version != self.normalization_version:
            return (
                TrustState.STALE,
                (
                    f"签名使用内容规范化版本 {record.normalization_version}，"
                    f"当前版本为 {self.normalization_version}，需要重签"
                ),
            )
        return TrustState.TRUSTED, "签名有效"

    def verify_notebook(self, nb: t.Any, path: str = "", event: str = "read") -> TrustVerdict:
        """验证笔记本当前的信任状态。

        只对规范化后的内容做哈希比对，绝不执行笔记本的代码或输出内容。
        ``event`` 标记触发场景（read / save / checkpoint-restore /
        external-modification / copy / cross-directory-copy / resign）。
        """
        now = _utcnow()
        if int(nb.get("nbformat", 0) or 0) < 3:
            return TrustVerdict(
                state=TrustState.UNKNOWN,
                trusted=False,
                reason="nbformat 版本过低，不参与信任判定",
                event=event,
            )
        content_hashes = self._content_hashes(nb)
        content_hash = content_hashes[self.normalization_version]
        external_change = False
        if path:
            previous = self.store.get_path_state(path)
            if previous is not None and previous != content_hash:
                external_change = True
                self.log.warning("检测到 %s 在内容管理器之外被修改，重新判断信任继承", path)
        records: dict[tuple[str, str], SignatureRecord] = {}
        for h in content_hashes.values():
            for record in self.store.records_for_hash(h):
                records.setdefault((record.digest, record.algorithm), record)
        best: TrustVerdict | None = None
        for record in records.values():
            secret = self._secrets.get(record.key_generation)
            if secret is None:
                continue
            if not self._digests_equal(nb, record, secret):
                continue
            state, reason = self._evaluate_record(record, now)
            verdict = TrustVerdict(
                state=state,
                trusted=state == TrustState.TRUSTED,
                reason=reason,
                record=record,
                event=event,
                external_change=external_change,
            )
            if best is None or _STATE_PRIORITY[state] < _STATE_PRIORITY[best.state]:
                best = verdict
            elif (
                state == best.state
                and best.record is not None
                and record.key_generation > best.record.key_generation
            ):
                # 同一状态下报告更新（更高代际）的签名记录
                best = verdict
        if best is None:
            best = TrustVerdict(
                state=TrustState.UNKNOWN,
                trusted=False,
                reason="没有找到与内容匹配的签名记录",
                event=event,
                external_change=external_change,
            )
        if path:
            self.store.set_path_state(path, content_hash)
        return best

    def _digests_equal(self, nb: t.Any, record: SignatureRecord, secret: bytes) -> bool:
        """用记录所属代际的密钥重算摘要并与记录比对。"""
        if record.normalization_version not in SUPPORTED_NORMALIZATION_VERSIONS:
            return False
        hmac = HMAC(secret, digestmod=getattr(hashlib, record.algorithm))
        for b in normalize_notebook(nb, record.normalization_version):
            hmac.update(b)
        return hmac.hexdigest() == record.digest

    def reevaluate(self, nb: t.Any, path: str = "", event: str = "read") -> TrustVerdict:
        """在另存、恢复检查点、外部修改、跨目录复制之后重新判断信任继承。"""
        verdict = self.verify_notebook(nb, path=path, event=event)
        log = self.log.warning if not verdict.trusted else self.log.info
        log(
            "信任继承重判[%s]：%s -> %s（%s）",
            event,
            path,
            verdict.state,
            verdict.reason,
        )
        return verdict

    # ------------------------------------------------------------------
    # 可中断续跑的重签任务
    # ------------------------------------------------------------------

    def begin_resign(self, paths: t.Iterable[str], task_id: str | None = None) -> ResignTask:
        """登记一个重签任务：把历史签名迁移到当前密钥代际与规范化版本。"""
        task_id = task_id or uuid.uuid4().hex
        self.store.create_resign_task(task_id, list(paths), _utcnow())
        return ResignTask(self, task_id)

    def resume_resign(self, task_id: str) -> ResignTask:
        """按任务 ID 继续一个此前中断的重签任务。"""
        if task_id not in self.store.resign_task_ids():
            msg = f"未知的重签任务: {task_id}"
            raise KeyError(msg)
        return ResignTask(self, task_id)

    def list_resign_tasks(self) -> list[str]:
        """列出全部重签任务 ID。"""
        return self.store.resign_task_ids()

    def _resign_one(self, nb: t.Any, path: str) -> tuple[str, str]:
        """处理一个重签条目，返回 (状态, 说明)。"""
        verdict = self.verify_notebook(nb, path=path, event="resign")
        if verdict.state == TrustState.REVOKED:
            # 已经撤销的签名不得恢复为可信
            return "skipped-revoked", verdict.reason
        if verdict.trusted:
            record = verdict.record
            if (
                record is not None
                and record.key_generation == self._current
                and record.normalization_version == self.normalization_version
            ):
                return "skipped-current", "已是当前密钥代际与规范化版本"
            new_record = self.sign_notebook(nb, path)
            if new_record is None:
                return "skipped-revoked", "内容已被撤销，拒绝重签"
            return "resigned", f"已用第 {new_record.key_generation} 代密钥重签"
        return "skipped-untrusted", verdict.reason

    def close(self) -> None:
        """关闭信任存储。"""
        self.store.close()


class ResignTask:
    """一个可中断、可续跑的重签任务。

    进度逐条持久化在信任存储中：中断（异常、进程退出或 ``interrupt()``）
    之后可用 ``TrustManager.resume_resign(task_id)`` 继续，已完成的条目
    不会重复处理，已撤销的签名不会被恢复为可信。
    """

    def __init__(self, manager: TrustManager, task_id: str):
        self.manager = manager
        self.task_id = task_id
        self._interrupted = False

    def interrupt(self) -> None:
        """请求中断；当前条目处理完后停止。"""
        self._interrupted = True

    def status(self) -> dict[str, int]:
        """按状态统计条目数量。"""
        counts: dict[str, int] = {}
        for item in self.manager.store.resign_items(self.task_id):
            counts[item["status"]] = counts.get(item["status"], 0) + 1
        return counts

    @property
    def pending_paths(self) -> list[str]:
        """尚未处理的条目路径。"""
        return [item["path"] for item in self.manager.store.resign_items(self.task_id, "pending")]

    @staticmethod
    def _default_loader(path: str) -> t.Any:
        with Path(path).open(encoding="utf-8") as f:
            return nbformat.read(f, nbformat.NO_CONVERT)

    def run(self, loader: t.Callable[[str], t.Any] | None = None) -> dict[str, int]:
        """逐条处理待办条目；中断时抛出 ResignInterrupted，可续跑。"""
        loader = loader or self._default_loader
        for item in self.manager.store.resign_items(self.task_id, "pending"):
            if self._interrupted:
                raise ResignInterrupted(self.task_id)
            path = item["path"]
            try:
                nb = loader(path)
                status, detail = self.manager._resign_one(nb, path)
            except ResignInterrupted:
                raise
            except Exception as e:  # noqa: BLE001 - 单个文件失败不阻断整个任务
                self.manager.log.warning("重签条目 %s 处理失败: %s", path, e, exc_info=True)
                status, detail = "error", str(e)
            self.manager.store.update_resign_item(self.task_id, path, status, detail)
        return self.status()
