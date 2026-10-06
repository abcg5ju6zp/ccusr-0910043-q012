"""分代际的笔记本签名信任管理。

在 nbformat 的签名模型之上扩展：

* 签名记录携带**密钥代际**、**签发时间**和**内容规范化版本**，
  安全人员可以据此区分签名产生于密钥泄露之前还是之后。
* 轮换密钥时可设置**验证窗口**（``verify_until``）与**撤销时点**
  （``revoke_before``）：撤销时点之前由旧密钥签发的记录在窗口关闭前
  仍可验证；撤销时点之后的一律不可信。
* 另存、恢复检查点、外部修改、跨目录复制时，由 ContentsManager
  重新判断信任继承（见 ``manager.py`` 中的钩子）。
* 验证过程只对内容做纯数据哈希，**不执行、不渲染输出内容**。
* 提供可中断、可续跑的批量重签任务；已撤销的签名永远不会被
  重新恢复为可信。
"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
from __future__ import annotations

import json
import logging
import os
import typing as t
from base64 import b64decode, b64encode, encodebytes
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hmac import HMAC
from pathlib import Path

import nbformat
from nbformat import sign
from traitlets import Unicode, default

# ---------------------------------------------------------------------------
# 信任判定原因
# ---------------------------------------------------------------------------

TRUSTED = "trusted"
UNSIGNED = "unsigned"
SIGNATURE_REVOKED = "signature-revoked"
KEY_GENERATION_REVOKED = "key-generation-revoked"
SIGNED_AFTER_REVOCATION_POINT = "signed-after-revocation-point"
VERIFICATION_WINDOW_EXPIRED = "verification-window-expired"
UNKNOWN_ISSUANCE_TIME = "unknown-issuance-time"
UNKNOWN_KEY_GENERATION = "unknown-key-generation"
UNSUPPORTED_NORMALIZATION = "unsupported-normalization"
UNSUPPORTED_NBFORMAT = "unsupported-nbformat"

#: 与撤销相关的判定原因（批量重签任务与自动签名都必须跳过）
REVOCATION_REASONS = frozenset(
    {SIGNATURE_REVOKED, KEY_GENERATION_REVOKED, SIGNED_AFTER_REVOCATION_POINT}
)

#: 阻止保存时自动继承信任（自动重签）的判定原因；
#: 显式的 trust_notebook 操作不受此限制。
BLOCKED_INHERIT_REASONS = frozenset(
    {
        SIGNATURE_REVOKED,
        KEY_GENERATION_REVOKED,
        SIGNED_AFTER_REVOCATION_POINT,
        VERIFICATION_WINDOW_EXPIRED,
        UNKNOWN_ISSUANCE_TIME,
    }
)

#: 笔记本 metadata 中放置签名提示（密钥代际/签发时间/规范化版本）的键。
#: 该提示只是便于排查的信息，信任判定以签名存储中的记录为准；
#: 计算摘要时该键会被排除。
SIGNATURE_RECORD_METADATA_KEY = "signature_record"

CURRENT_NORMALIZATION_VERSION = "v1"


class UnknownNormalizationVersion(ValueError):
    """请求的内容规范化版本未注册。"""


class UnknownKeyGeneration(ValueError):
    """请求的密钥代际不在密钥环中（或密钥材料已清除）。"""


# ---------------------------------------------------------------------------
# 内容规范化
# ---------------------------------------------------------------------------


@contextmanager
def signature_metadata_removed(nb):
    """临时移除签名相关 metadata，用于计算内容摘要。

    同时排除 nbformat 的 ``metadata.signature`` 与本模块的
    ``metadata.signature_record`` 提示，保证提示信息不影响摘要。
    """
    metadata = nb["metadata"]
    saved_signature = metadata.pop("signature", None)
    saved_record = metadata.pop(SIGNATURE_RECORD_METADATA_KEY, None)
    try:
        yield
    finally:
        if saved_signature is not None:
            metadata["signature"] = saved_signature
        if saved_record is not None:
            metadata[SIGNATURE_RECORD_METADATA_KEY] = saved_record


def normalize_v1(nb):
    """v1 规范化：nbformat 的规范字节流（排除签名 metadata）。

    与 nbformat 历史行为完全一致，因此旧签名记录仍可验证。
    只读取内容做哈希，不执行任何输出。
    """
    with signature_metadata_removed(nb):
        yield from sign.yield_everything(nb)


_NORMALIZERS: dict[str, t.Callable[[t.Any], t.Iterable[bytes]]] = {}


def register_normalization(version: str, normalizer: t.Callable[[t.Any], t.Iterable[bytes]]):
    """注册一个内容规范化版本。验证时会按记录中的版本选择对应实现。"""
    _NORMALIZERS[version] = normalizer


def registered_normalizations() -> list[str]:
    """返回已注册的规范化版本列表。"""
    return sorted(_NORMALIZERS)


register_normalization("v1", normalize_v1)


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


def _parse_dt(value: t.Any) -> datetime | None:
    """把 ISO 字符串/datetime 解析为带时区的 datetime；None 原样返回。"""
    if value is None or isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value))
    if dt is not None and dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


@dataclass
class KeyGeneration:
    """一个密钥代际及其轮换策略。"""

    generation: int
    created_at: datetime
    secret: bytes | None = None
    verify_until: datetime | None = None
    revoke_before: datetime | None = None
    revoked_at: datetime | None = None
    revoked_reason: str = ""
    rotation_reason: str = ""

    def to_dict(self, include_secret: bool = True) -> dict[str, t.Any]:
        """序列化；include_secret=False 用于对外状态展示。"""
        data: dict[str, t.Any] = {
            "generation": self.generation,
            "created_at": _iso(self.created_at),
            "verify_until": _iso(self.verify_until),
            "revoke_before": _iso(self.revoke_before),
            "revoked_at": _iso(self.revoked_at),
            "revoked_reason": self.revoked_reason,
            "rotation_reason": self.rotation_reason,
            "has_secret": self.secret is not None,
        }
        if include_secret:
            data["secret"] = b64encode(self.secret).decode("ascii") if self.secret else None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, t.Any]) -> KeyGeneration:
        """反序列化。"""
        secret = data.get("secret")
        return cls(
            generation=int(data["generation"]),
            created_at=_parse_dt(data.get("created_at")) or datetime.now(timezone.utc),
            secret=b64decode(secret) if secret else None,
            verify_until=_parse_dt(data.get("verify_until")),
            revoke_before=_parse_dt(data.get("revoke_before")),
            revoked_at=_parse_dt(data.get("revoked_at")),
            revoked_reason=data.get("revoked_reason") or "",
            rotation_reason=data.get("rotation_reason") or "",
        )


@dataclass
class SignatureRecord:
    """一条签名记录：摘要 + 密钥代际 + 签发时间 + 规范化版本 + 撤销信息。"""

    digest: str
    algorithm: str
    key_generation: int | None = None
    issued_at: datetime | None = None
    normalization_version: str = CURRENT_NORMALIZATION_VERSION
    revoked_at: datetime | None = None
    revoked_reason: str = ""

    def to_dict(self) -> dict[str, t.Any]:
        """序列化。"""
        return {
            "digest": self.digest,
            "algorithm": self.algorithm,
            "key_generation": self.key_generation,
            "issued_at": _iso(self.issued_at),
            "normalization_version": self.normalization_version,
            "revoked_at": _iso(self.revoked_at),
            "revoked_reason": self.revoked_reason,
        }


@dataclass
class TrustDecision:
    """一次信任判定的结果。"""

    trusted: bool
    reason: str
    record: SignatureRecord | None = None
    key_generation: int | None = None

    def to_dict(self) -> dict[str, t.Any]:
        """序列化。"""
        return {
            "trusted": self.trusted,
            "reason": self.reason,
            "key_generation": self.key_generation,
            "record": self.record.to_dict() if self.record else None,
        }


# ---------------------------------------------------------------------------
# 密钥环
# ---------------------------------------------------------------------------


class Keyring:
    """文件保存的分代际密钥环。

    每个代际保存独立的签名密钥与轮换策略（验证窗口、撤销时点、
    撤销标记）。旧代际的密钥在验证窗口关闭后会被清除（保留不含
    密钥材料的墓碑元数据用于审计）；仅被撤销但窗口未关闭的代际
    保留密钥，以便验证时给出准确的不可信原因。
    """

    def __init__(
        self,
        generations: dict[int, KeyGeneration],
        current: int,
        path: Path | None = None,
        log: logging.Logger | None = None,
    ):
        """初始化密钥环。"""
        self.generations = generations
        self.current = current
        self.path = path
        self.log = log or logging.getLogger(__name__)

    @classmethod
    def load(
        cls,
        path: str | os.PathLike | None,
        legacy_secret_file: str | None = None,
        log: logging.Logger | None = None,
    ) -> Keyring:
        """加载密钥环；不存在时从 legacy notebook_secret 导入或新建。"""
        log = log or logging.getLogger(__name__)
        ring_path = Path(path) if path else None
        if ring_path is not None and ring_path.exists():
            data = json.loads(ring_path.read_text(encoding="utf-8"))
            generations = {
                int(g): KeyGeneration.from_dict(item) for g, item in data["generations"].items()
            }
            ring = cls(generations, int(data["current"]), path=ring_path, log=log)
            ring.purge_expired()
            return ring

        generations: dict[int, KeyGeneration] = {}
        now = datetime.now(timezone.utc)
        if legacy_secret_file and Path(legacy_secret_file).exists():
            secret = Path(legacy_secret_file).read_bytes()
            try:
                created = datetime.fromtimestamp(
                    Path(legacy_secret_file).stat().st_mtime, tz=timezone.utc
                )
            except OSError:
                created = now
            generations[1] = KeyGeneration(generation=1, created_at=created, secret=secret)
            log.info("Imported legacy notebook secret from %s as key generation 1", legacy_secret_file)
        else:
            generations[1] = KeyGeneration(
                generation=1, created_at=now, secret=encodebytes(os.urandom(1024))
            )
        ring = cls(generations, 1, path=ring_path, log=log)
        ring.save()
        return ring

    def save(self):
        """持久化密钥环（原子写入，权限 0600）。"""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "current": self.current,
            "generations": {
                str(g): gen.to_dict(include_secret=True) for g, gen in self.generations.items()
            },
        }
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(tmp_path, self.path)
        try:
            self.path.chmod(0o600)
        except OSError:
            self.log.warning("Could not set permissions on %s", self.path)

    def get(self, generation: int) -> KeyGeneration | None:
        """按代际号取密钥代际。"""
        return self.generations.get(generation)

    def secret_for(self, generation: int) -> bytes | None:
        """取某代际的密钥材料；已清除则返回 None。"""
        gen = self.get(generation)
        return gen.secret if gen else None

    def rotate(
        self,
        verify_old_until: datetime | None = None,
        revoke_before: datetime | None = None,
        reason: str = "",
        now: datetime | None = None,
    ) -> int:
        """轮换密钥：为当前代际设置验证窗口/撤销时点，并生成新代际。

        ``revoke_before`` 是泄露时点估计：该时点之后由旧密钥签发的
        记录视为不可信；``verify_old_until`` 是验证窗口的终点，超过后
        旧代际不再用于验证，其密钥材料被清除。
        """
        now = now or datetime.now(timezone.utc)
        old = self.generations[self.current]
        if verify_old_until is not None:
            old.verify_until = verify_old_until
        if revoke_before is not None:
            old.revoke_before = revoke_before
        if reason:
            old.rotation_reason = reason
        new_generation = max(self.generations) + 1
        self.generations[new_generation] = KeyGeneration(
            generation=new_generation,
            created_at=now,
            secret=encodebytes(os.urandom(1024)),
        )
        self.current = new_generation
        self.purge_expired(now=now)
        self.save()
        self.log.warning(
            "Rotated notebook signing key to generation %d (verify old until %s, revoke before %s): %s",
            new_generation,
            _iso(verify_old_until),
            _iso(revoke_before),
            reason or "no reason given",
        )
        return new_generation

    def revoke_generation(self, generation: int, reason: str = "", now: datetime | None = None):
        """撤销整个代际：该代际签发的所有记录不再可信。"""
        gen = self.generations[generation]
        gen.revoked_at = now or datetime.now(timezone.utc)
        gen.revoked_reason = reason
        self.purge_expired(now=gen.revoked_at)
        self.save()

    def purge_expired(self, now: datetime | None = None):
        """清除验证窗口已关闭代际的密钥材料（保留元数据）。

        已撤销但窗口未关闭的代际保留密钥：验证时需要用它计算
        摘要才能给出“代际已撤销”的准确判定，而不是笼统的未知签名。
        """
        now = now or datetime.now(timezone.utc)
        for gen in self.generations.values():
            if gen.secret is None or gen.generation == self.current:
                continue
            if gen.verify_until is not None and gen.verify_until <= now:
                gen.secret = None
                self.log.info(
                    "Purged secret of notebook signing key generation %d "
                    "(verification window closed)",
                    gen.generation,
                )

    def status(self) -> dict[str, t.Any]:
        """对外状态（不含密钥材料）。"""
        return {
            "current": self.current,
            "generations": [
                self.generations[g].to_dict(include_secret=False)
                for g in sorted(self.generations)
            ],
        }


# ---------------------------------------------------------------------------
# 签名存储
# ---------------------------------------------------------------------------


class GenerationalSignatureStore(sign.SQLiteSignatureStore):
    """携带代际/签发时间/规范化版本/撤销信息的 SQLite 签名存储。

    在 nbformat 的 nbsignatures 表上增量迁移出新列，并新增
    resign_state 表保存批量重签任务的进度。
    """

    def init_db(self, db):
        """初始化并迁移数据库结构。"""
        super().init_db(db)
        columns = {row[1] for row in db.execute("PRAGMA table_info(nbsignatures)")}
        additions = {
            "key_generation": "ALTER TABLE nbsignatures ADD COLUMN key_generation integer",
            "issued_at": "ALTER TABLE nbsignatures ADD COLUMN issued_at text",
            "normalization_version": (
                "ALTER TABLE nbsignatures ADD COLUMN normalization_version text DEFAULT 'v1'"
            ),
            "revoked_at": "ALTER TABLE nbsignatures ADD COLUMN revoked_at text",
            "revoked_reason": "ALTER TABLE nbsignatures ADD COLUMN revoked_reason text",
        }
        for column, ddl in additions.items():
            if column not in columns:
                db.execute(ddl)
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS resign_state
            (
                task_id text PRIMARY KEY,
                state text,
                updated_at text
            )"""
        )
        db.commit()

    # -- 记录读写 -----------------------------------------------------------

    def _row_to_record(self, row) -> SignatureRecord:
        """把数据库行转换为 SignatureRecord。"""
        (algorithm, digest, key_generation, issued_at, normalization_version,
         revoked_at, revoked_reason) = row
        return SignatureRecord(
            digest=digest,
            algorithm=algorithm,
            key_generation=key_generation,
            issued_at=_parse_dt(issued_at),
            normalization_version=normalization_version or CURRENT_NORMALIZATION_VERSION,
            revoked_at=_parse_dt(revoked_at),
            revoked_reason=revoked_reason or "",
        )

    # 注意：不能 SELECT last_seen——连接启用了 PARSE_DECLTYPES，
    # 内置 timestamp 转换器无法解析带时区的 ISO 字符串。
    _RECORD_COLUMNS = (
        "algorithm, signature, key_generation, issued_at, "
        "normalization_version, revoked_at, revoked_reason"
    )

    def store_record(self, record: SignatureRecord) -> bool:
        """写入签名记录。

        已撤销的记录具有粘性：同一摘要被撤销后，重复写入不会清除
        撤销标记（返回 False）。要恢复必须先显式 clear_revocation。
        """
        if self.db is None:
            return False
        existing = self.get_record(record.digest, record.algorithm, update_last_seen=False)
        if existing is not None and existing.revoked_at is not None:
            self.log.warning(
                "Refusing to re-store revoked signature %s (%s)",
                record.digest[:12],
                existing.revoked_reason or "no reason recorded",
            )
            return False
        now = datetime.now(tz=timezone.utc).isoformat()
        if existing is None:
            self.db.execute(
                """
                INSERT INTO nbsignatures
                    (algorithm, signature, last_seen, key_generation, issued_at,
                     normalization_version, revoked_at, revoked_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.algorithm,
                    record.digest,
                    now,
                    record.key_generation,
                    _iso(record.issued_at),
                    record.normalization_version,
                    _iso(record.revoked_at),
                    record.revoked_reason,
                ),
            )
        else:
            self.db.execute(
                """
                UPDATE nbsignatures SET last_seen = ?, key_generation = ?,
                    issued_at = ?, normalization_version = ?
                WHERE algorithm = ? AND signature = ?
                """,
                (
                    now,
                    record.key_generation,
                    _iso(record.issued_at),
                    record.normalization_version,
                    record.algorithm,
                    record.digest,
                ),
            )
        self.db.commit()
        (n,) = self.db.execute("SELECT Count(*) FROM nbsignatures").fetchone()
        if n > self.cache_size:
            self.cull_db()
        return True

    def store_signature(self, digest, algorithm):
        """兼容 nbformat 的两参数接口：写入无代际信息的记录。"""
        self.store_record(SignatureRecord(digest=digest, algorithm=algorithm))

    def get_record(
        self, digest: str, algorithm: str, update_last_seen: bool = True
    ) -> SignatureRecord | None:
        """按摘要查询签名记录。"""
        if self.db is None:
            return None
        row = self.db.execute(
            f"SELECT {self._RECORD_COLUMNS} FROM nbsignatures WHERE "  # noqa: S608
            "algorithm = ? AND signature = ? ORDER BY id DESC LIMIT 1",
            (algorithm, digest),
        ).fetchone()
        if row is None:
            return None
        if update_last_seen:
            self.db.execute(
                "UPDATE nbsignatures SET last_seen = ? WHERE algorithm = ? AND signature = ?",
                (datetime.now(tz=timezone.utc).isoformat(), algorithm, digest),
            )
            self.db.commit()
        return self._row_to_record(row)

    def check_signature(self, digest, algorithm):
        """兼容 nbformat 的布尔接口（不做代际策略判定）。"""
        return self.get_record(digest, algorithm) is not None

    def iter_records(self) -> t.Iterator[SignatureRecord]:
        """遍历全部签名记录。"""
        if self.db is None:
            return
        rows = self.db.execute(
            f"SELECT {self._RECORD_COLUMNS} FROM nbsignatures ORDER BY id"  # noqa: S608
        ).fetchall()
        for row in rows:
            yield self._row_to_record(row)

    def revoke_signature(
        self, digest: str, algorithm: str, when: datetime | None = None, reason: str = ""
    ) -> int:
        """撤销指定摘要的签名记录；返回更新的行数。"""
        if self.db is None:
            return 0
        when = when or datetime.now(timezone.utc)
        cursor = self.db.execute(
            """
            UPDATE nbsignatures SET revoked_at = ?, revoked_reason = ?
            WHERE algorithm = ? AND signature = ? AND revoked_at IS NULL
            """,
            (when.isoformat(), reason, algorithm, digest),
        )
        self.db.commit()
        return cursor.rowcount

    def clear_revocation(self, digest: str, algorithm: str) -> int:
        """显式解除撤销（管理操作）；返回更新的行数。"""
        if self.db is None:
            return 0
        cursor = self.db.execute(
            "UPDATE nbsignatures SET revoked_at = NULL, revoked_reason = NULL "
            "WHERE algorithm = ? AND signature = ?",
            (algorithm, digest),
        )
        self.db.commit()
        return cursor.rowcount

    def backfill_records(self, generation: int) -> int:
        """把迁移前缺少代际信息的旧记录归属到指定代际。"""
        if self.db is None:
            return 0
        cursor = self.db.execute(
            "UPDATE nbsignatures SET key_generation = ? WHERE key_generation IS NULL",
            (generation,),
        )
        self.db.commit()
        return cursor.rowcount

    # -- 重签任务进度 --------------------------------------------------------

    def get_resign_state(self, task_id: str) -> dict[str, t.Any] | None:
        """读取重签任务进度。"""
        if self.db is None:
            return None
        row = self.db.execute(
            "SELECT state FROM resign_state WHERE task_id = ?", (task_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def set_resign_state(self, task_id: str, state: dict[str, t.Any]):
        """保存重签任务进度。"""
        if self.db is None:
            return
        self.db.execute(
            "INSERT OR REPLACE INTO resign_state (task_id, state, updated_at) VALUES (?, ?, ?)",
            (task_id, json.dumps(state), datetime.now(timezone.utc).isoformat()),
        )
        self.db.commit()


class MemoryGenerationalSignatureStore(sign.SignatureStore):
    """内存版分代际签名存储（无 SQLite 时的降级实现）。"""

    cache_size = 65535

    def __init__(self):
        """初始化内存存储。"""
        self.records: OrderedDict[tuple[str, str], SignatureRecord] = OrderedDict()
        self.resign_states: dict[str, dict[str, t.Any]] = {}

    def _maybe_cull(self):
        """超过容量时淘汰最旧的 25% 记录。"""
        if len(self.records) < self.cache_size:
            return
        for _ in range(len(self.records) // 4):
            self.records.popitem(last=False)

    def store_record(self, record: SignatureRecord) -> bool:
        """写入签名记录；已撤销记录保持粘性。"""
        key = (record.digest, record.algorithm)
        existing = self.records.get(key)
        if existing is not None and existing.revoked_at is not None:
            return False
        self.records.pop(key, None)
        self.records[key] = record
        self._maybe_cull()
        return True

    def store_signature(self, digest, algorithm):
        """兼容 nbformat 的两参数接口。"""
        self.store_record(SignatureRecord(digest=digest, algorithm=algorithm))

    def get_record(
        self, digest: str, algorithm: str, update_last_seen: bool = True
    ) -> SignatureRecord | None:
        """按摘要查询签名记录。"""
        record = self.records.get((digest, algorithm))
        if record is not None and update_last_seen:
            self.records.move_to_end((digest, algorithm))
        return record

    def check_signature(self, digest, algorithm):
        """兼容 nbformat 的布尔接口。"""
        return self.get_record(digest, algorithm) is not None

    def remove_signature(self, digest, algorithm):
        """删除签名记录。"""
        self.records.pop((digest, algorithm), None)

    def iter_records(self) -> t.Iterator[SignatureRecord]:
        """遍历全部签名记录。"""
        yield from list(self.records.values())

    def revoke_signature(
        self, digest: str, algorithm: str, when: datetime | None = None, reason: str = ""
    ) -> int:
        """撤销指定摘要的签名记录。"""
        record = self.records.get((digest, algorithm))
        if record is None or record.revoked_at is not None:
            return 0
        record.revoked_at = when or datetime.now(timezone.utc)
        record.revoked_reason = reason
        return 1

    def clear_revocation(self, digest: str, algorithm: str) -> int:
        """显式解除撤销。"""
        record = self.records.get((digest, algorithm))
        if record is None or record.revoked_at is None:
            return 0
        record.revoked_at = None
        record.revoked_reason = ""
        return 1

    def backfill_records(self, generation: int) -> int:
        """把缺少代际信息的旧记录归属到指定代际。"""
        n = 0
        for record in self.records.values():
            if record.key_generation is None:
                record.key_generation = generation
                n += 1
        return n

    def get_resign_state(self, task_id: str) -> dict[str, t.Any] | None:
        """读取重签任务进度。"""
        return self.resign_states.get(task_id)

    def set_resign_state(self, task_id: str, state: dict[str, t.Any]):
        """保存重签任务进度。"""
        self.resign_states[task_id] = dict(state)


# ---------------------------------------------------------------------------
# 分代际 Notary
# ---------------------------------------------------------------------------


class GenerationalNotebookNotary(sign.NotebookNotary):
    """支持密钥代际、验证窗口与撤销的笔记本签名器。

    验证是纯函数：只对规范化后的内容计算 HMAC 并查询签名存储，
    不执行、不渲染笔记本中的任何输出内容。
    """

    normalization_version = Unicode(
        CURRENT_NORMALIZATION_VERSION,
        help="""The content normalization version used for new signatures.""",
    ).tag(config=True)

    keyring_file = Unicode(
        help="""The file where the key ring of signing key generations is stored.
        Empty string keeps the key ring in memory only."""
    ).tag(config=True)

    @default("keyring_file")
    def _keyring_file_default(self):
        if not self.data_dir:
            return ""
        return str(Path(self.data_dir) / "notebook_keyring.json")

    @default("store_factory")
    def _store_factory_default(self):
        def factory():
            if sign.sqlite3 is None:
                self.log.warning("Missing SQLite3, all notebooks will be untrusted!")  # type: ignore[unreachable]
                return MemoryGenerationalSignatureStore()
            return GenerationalSignatureStore(self.db_file)

        return factory

    def __init__(self, **kwargs):
        """初始化：加载密钥环并迁移旧签名记录。"""
        super().__init__(**kwargs)
        self.keyring = Keyring.load(
            self.keyring_file or None,
            legacy_secret_file=self.secret_file or None,
            log=self.log,
        )
        self._backfill_store()

    @default("secret")
    def _secret_default(self):
        secret = self.keyring.secret_for(self.keyring.current)
        return secret if secret is not None else b""

    def _backfill_store(self):
        """把旧库中缺少代际信息的记录归属到当前唯一代际。"""
        backfill = getattr(self.store, "backfill_records", None)
        if backfill is None or len(self.keyring.generations) != 1:
            return
        n = backfill(self.keyring.current)
        if n:
            self.log.info(
                "Backfilled %d legacy signature record(s) to key generation %d",
                n,
                self.keyring.current,
            )

    # -- 基础信息 ------------------------------------------------------------

    @property
    def current_generation(self) -> int:
        """当前签名使用的密钥代际。"""
        return self.keyring.current

    def _now(self) -> datetime:
        """当前时间（单独的方法便于测试注入）。"""
        return datetime.now(timezone.utc)

    # -- 摘要计算 --------------------------------------------------------------

    def compute_signature(self, nb, normalization_version=None, generation=None):
        """计算笔记本内容摘要。

        只对内容做哈希：输出内容作为数据参与摘要，绝不执行。
        ``normalization_version`` 选择内容规范化实现，``generation``
        选择使用哪一代密钥。
        """
        version = normalization_version or self.normalization_version
        normalizer = _NORMALIZERS.get(version)
        if normalizer is None:
            raise UnknownNormalizationVersion(version)
        generation = self.keyring.current if generation is None else generation
        secret = self.keyring.secret_for(generation)
        if secret is None:
            raise UnknownKeyGeneration(generation)
        hmac = HMAC(secret, digestmod=self.digestmod)
        for b in normalizer(nb):
            hmac.update(b)
        return hmac.hexdigest()

    # -- 签名与验证 ------------------------------------------------------------

    def sign(self, nb):
        """用当前代际密钥签名笔记本，并记录代际/签发时间/规范化版本。"""
        if nb.nbformat < 3:
            return
        record = SignatureRecord(
            digest=self.compute_signature(nb),
            algorithm=self.algorithm,
            key_generation=self.keyring.current,
            issued_at=self._now(),
            normalization_version=self.normalization_version,
        )
        store_record = getattr(self.store, "store_record", None)
        if store_record is None:
            # 降级：外部配置的旧式存储
            self.store.store_signature(record.digest, record.algorithm)
            self._write_hint(nb, record)
            return
        if store_record(record):
            self._write_hint(nb, record)

    def unsign(self, nb):
        """确保笔记本不可信：删除所有代际/版本下匹配的记录。"""
        for generation, version in self._verification_candidates(None):
            try:
                digest = self.compute_signature(nb, normalization_version=version, generation=generation)
            except (UnknownKeyGeneration, UnknownNormalizationVersion):
                continue
            self.store.remove_signature(digest, self.algorithm)
        metadata = nb.get("metadata", {})
        metadata.pop(SIGNATURE_RECORD_METADATA_KEY, None)

    def check_signature(self, nb):
        """检查笔记本是否可信（含代际策略判定）。"""
        return self.evaluate_trust(nb).trusted

    def evaluate_trust(self, nb, now: datetime | None = None) -> TrustDecision:
        """对笔记本做完整的信任判定。

        依次尝试候选的（密钥代际, 规范化版本）组合计算摘要并查询
        签名存储；任一记录通过策略判定即可信。撤销类原因优先于
        一般性原因返回。整个过程不执行笔记本内容。
        """
        now = now or self._now()
        if nb.nbformat < 3:
            return TrustDecision(False, UNSUPPORTED_NBFORMAT)
        get_record = getattr(self.store, "get_record", None)
        if get_record is None:
            # 降级：旧式存储只有布尔接口
            trusted = self.store.check_signature(self.compute_signature(nb), self.algorithm)
            return TrustDecision(trusted, TRUSTED if trusted else UNSIGNED)

        hint = self._read_hint(nb)
        best = TrustDecision(False, UNSIGNED)
        for generation, version in self._verification_candidates(hint):
            try:
                digest = self.compute_signature(
                    nb, normalization_version=version, generation=generation
                )
            except (UnknownKeyGeneration, UnknownNormalizationVersion):
                continue
            record = get_record(digest, self.algorithm)
            if record is None:
                continue
            decision = self.evaluate_record(record, now)
            if decision.trusted:
                return decision
            if best.reason not in REVOCATION_REASONS and (
                decision.reason in REVOCATION_REASONS or best.reason == UNSIGNED
            ):
                best = decision
        return best

    def evaluate_record(self, record: SignatureRecord, now: datetime | None = None) -> TrustDecision:
        """依据轮换策略判定一条签名记录是否可信。"""
        now = now or self._now()
        generation = record.key_generation
        if record.revoked_at is not None:
            return TrustDecision(False, SIGNATURE_REVOKED, record, generation)
        if record.normalization_version not in _NORMALIZERS:
            return TrustDecision(False, UNSUPPORTED_NORMALIZATION, record, generation)
        if generation is not None:
            gen = self.keyring.get(generation)
            if gen is None:
                return TrustDecision(False, UNKNOWN_KEY_GENERATION, record, generation)
            if gen.revoked_at is not None:
                return TrustDecision(False, KEY_GENERATION_REVOKED, record, generation)
            if gen.revoke_before is not None:
                if record.issued_at is None:
                    # 无法证明签名产生于泄露时点之前，按不可信处理
                    return TrustDecision(False, UNKNOWN_ISSUANCE_TIME, record, generation)
                if record.issued_at >= gen.revoke_before:
                    return TrustDecision(False, SIGNED_AFTER_REVOCATION_POINT, record, generation)
            if gen.verify_until is not None and now > gen.verify_until:
                return TrustDecision(False, VERIFICATION_WINDOW_EXPIRED, record, generation)
        return TrustDecision(True, TRUSTED, record, generation)

    # -- 轮换与撤销 ------------------------------------------------------------

    def rotate_key(
        self,
        verify_old_until: datetime | None = None,
        revoke_before: datetime | None = None,
        reason: str = "",
    ) -> int:
        """轮换签名密钥，返回新代际号。

        ``verify_old_until``：旧代际签名的验证窗口终点；
        ``revoke_before``：撤销时点，该时点及之后由旧密钥签发的
        记录视为不可信。
        """
        new_generation = self.keyring.rotate(
            verify_old_until=verify_old_until,
            revoke_before=revoke_before,
            reason=reason,
            now=self._now(),
        )
        secret = self.keyring.secret_for(new_generation)
        if secret is not None:
            self.secret = secret
        return new_generation

    def revoke_generation(self, generation: int, reason: str = ""):
        """撤销整个密钥代际。"""
        self.keyring.revoke_generation(generation, reason=reason, now=self._now())

    def revoke_signature(self, nb_or_digest, reason: str = "") -> int:
        """撤销单个签名（按笔记本内容或摘要）；返回撤销的记录数。"""
        revoke = getattr(self.store, "revoke_signature", None)
        if revoke is None:
            return 0
        when = self._now()
        if isinstance(nb_or_digest, str):
            return revoke(nb_or_digest, self.algorithm, when, reason)
        count = 0
        for generation, version in self._verification_candidates(self._read_hint(nb_or_digest)):
            try:
                digest = self.compute_signature(
                    nb_or_digest, normalization_version=version, generation=generation
                )
            except (UnknownKeyGeneration, UnknownNormalizationVersion):
                continue
            count += revoke(digest, self.algorithm, when, reason)
        return count

    def clear_revocation(self, digest: str) -> int:
        """显式解除对指定摘要的撤销（管理操作）。"""
        clear = getattr(self.store, "clear_revocation", None)
        return clear(digest, self.algorithm) if clear else 0

    # -- 状态 ------------------------------------------------------------------

    def trust_status(self) -> dict[str, t.Any]:
        """对外展示的信任状态（不含密钥材料）。"""
        return {
            "current_generation": self.keyring.current,
            "normalization_version": self.normalization_version,
            "registered_normalizations": registered_normalizations(),
            "algorithm": self.algorithm,
            "keyring": self.keyring.status(),
        }

    # -- 内部工具 ----------------------------------------------------------------

    def _verification_candidates(self, hint: dict[str, t.Any] | None):
        """生成（代际, 规范化版本）候选对，提示信息优先。"""
        pairs: list[tuple[int, str]] = []
        if hint:
            hint_generation = hint.get("key_generation")
            hint_version = hint.get("normalization_version")
            if isinstance(hint_generation, int) and isinstance(hint_version, str):
                pairs.append((hint_generation, hint_version))
        versions = [self.normalization_version] + [
            v for v in registered_normalizations() if v != self.normalization_version
        ]
        generations = [self.keyring.current] + [
            g for g in sorted(self.keyring.generations) if g != self.keyring.current
        ]
        for generation in generations:
            for version in versions:
                pair = (generation, version)
                if pair not in pairs:
                    pairs.append(pair)
        return pairs

    def _write_hint(self, nb, record: SignatureRecord):
        """把签名提示写入笔记本 metadata（不参与摘要）。"""
        if nb.nbformat < 3:
            return
        nb["metadata"][SIGNATURE_RECORD_METADATA_KEY] = {
            "key_generation": record.key_generation,
            "issued_at": _iso(record.issued_at),
            "normalization_version": record.normalization_version,
            "algorithm": record.algorithm,
        }

    def _read_hint(self, nb) -> dict[str, t.Any] | None:
        """读取笔记本 metadata 中的签名提示。"""
        try:
            hint = nb.get("metadata", {}).get(SIGNATURE_RECORD_METADATA_KEY)
        except AttributeError:
            return None
        return hint if isinstance(hint, dict) else None


# ---------------------------------------------------------------------------
# 批量重签任务
# ---------------------------------------------------------------------------


def iter_notebooks(root_dir: str | os.PathLike, log: logging.Logger | None = None):
    """按路径顺序产出 (相对路径, 笔记本对象)，跳过隐藏目录与检查点目录。"""
    log = log or logging.getLogger(__name__)
    root = Path(root_dir)
    paths = sorted(
        p
        for p in root.rglob("*.ipynb")
        if not any(part.startswith(".") for part in p.relative_to(root).parts)
    )
    for path in paths:
        try:
            with path.open(encoding="utf-8") as f:
                nb = nbformat.read(f, nbformat.NO_CONVERT)
        except Exception as e:
            log.warning("Skipping unreadable notebook %s: %s", path, e)
            continue
        yield str(path.relative_to(root)), nb


class ResignTask:
    """把仍然有效的签名迁移到当前密钥代际的批量任务。

    每处理完一个笔记本就把进度写入签名存储，因此任务可以随时
    中断（异常、KeyboardInterrupt），再次运行会从断点继续。
    已撤销的签名永远不会被重新签名——只有当前判定为可信且
    不属于当前代际的记录才会被迁移。
    """

    def __init__(self, notary: GenerationalNotebookNotary, task_id: str = "default", log=None):
        """初始化重签任务。"""
        self.notary = notary
        self.task_id = task_id
        self.log = log or getattr(notary, "log", None) or logging.getLogger(__name__)

    def run(self, items, should_stop: t.Callable[[], bool] | None = None) -> dict[str, t.Any]:
        """处理 ``(key, notebook)`` 序列；key 必须按字典序稳定（如路径）。

        ``should_stop`` 返回 True 时合作式中断；进度已持久化，
        再次调用 run（相同 items）即可续跑。
        """
        store = self.notary.store
        state = store.get_resign_state(self.task_id) or {}
        last_key = state.get("last_key")
        stats: dict[str, t.Any] = {
            "resigned": 0,
            "skipped_current": 0,
            "skipped_revoked": 0,
            "skipped_untrusted": 0,
            "done": False,
        }
        for key, nb in items:
            if last_key is not None and key <= last_key:
                continue
            if should_stop is not None and should_stop():
                self.log.info("Resign task %s interrupted at %s", self.task_id, key)
                return stats
            self._process(key, nb, stats)
            last_key = key
            store.set_resign_state(self.task_id, {"last_key": key, "done": False})
        store.set_resign_state(self.task_id, {"last_key": last_key, "done": True})
        stats["done"] = True
        return stats

    def _process(self, key, nb, stats):
        """判定并迁移单个笔记本的签名。"""
        decision = self.notary.evaluate_trust(nb)
        record = decision.record
        if decision.trusted:
            if record is None or record.key_generation != self.notary.current_generation:
                self.notary.sign(nb)
                stats["resigned"] += 1
                self.log.debug("Re-signed %s under generation %d", key, self.notary.current_generation)
            else:
                stats["skipped_current"] += 1
        elif decision.reason in REVOCATION_REASONS or (record and record.revoked_at):
            # 已撤销的签名绝不恢复为可信
            stats["skipped_revoked"] += 1
            self.log.warning("Not re-signing revoked notebook %s (%s)", key, decision.reason)
        else:
            stats["skipped_untrusted"] += 1
