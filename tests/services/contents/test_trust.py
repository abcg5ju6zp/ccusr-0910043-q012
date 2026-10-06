"""分代际信任管理的回归测试。

覆盖：签名记录字段（密钥代际/签发时间/规范化版本）、轮换的验证窗口
与撤销时点、另存/检查点恢复/外部修改/跨目录复制的信任继承重估、
验证过程不执行输出内容、可中断续跑的重签任务、已撤销签名不被恢复。
"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from jupyter_core.utils import ensure_async
from nbformat.sign import NotebookNotary
from nbformat.v4 import new_code_cell, new_notebook, new_output

from jupyter_server.services.contents.filemanager import (
    AsyncFileContentsManager,
    FileContentsManager,
)
from jupyter_server.services.contents.trust import (
    KEY_GENERATION_REVOKED,
    SIGNATURE_REVOKED,
    SIGNED_AFTER_REVOCATION_POINT,
    UNKNOWN_ISSUANCE_TIME,
    UNSIGNED,
    VERIFICATION_WINDOW_EXPIRED,
    GenerationalNotebookNotary,
    ResignTask,
    iter_notebooks,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


class Clock:
    """可手动推进的时钟，注入 notary._now。"""

    def __init__(self, start=T0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, **kwargs):
        self.now += timedelta(**kwargs)


def make_nb(source="print(1)", trusted=True, outputs=None):
    """构造一个带可信代码单元格的笔记本。"""
    nb = new_notebook()
    cell = new_code_cell(source)
    cell.metadata.trusted = trusted
    if outputs:
        cell.outputs = outputs
    nb.cells.append(cell)
    return nb


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def notary(tmp_path, clock):
    notary_dir = tmp_path / "notary"
    notary_dir.mkdir()
    n = GenerationalNotebookNotary(data_dir=str(notary_dir))
    n._now = clock
    return n


@pytest.fixture(params=[FileContentsManager, AsyncFileContentsManager])
def cm(request, tmp_path, clock):
    root = tmp_path / "root"
    root.mkdir()
    notary_dir = tmp_path / "notary"
    notary_dir.mkdir()
    manager = request.param(root_dir=str(root))
    manager.notary = GenerationalNotebookNotary(data_dir=str(notary_dir))
    manager.notary._now = clock
    return manager


async def save_nb(cm, nb, path):
    """通过管理器保存笔记本模型。"""
    model = {"type": "notebook", "content": nb}
    return await ensure_async(cm.save(model, path))


async def get_nb(cm, path):
    model = await ensure_async(cm.get(path))
    return model["content"]


def trusted_flags(nb):
    return [cell.metadata.get("trusted") for cell in nb.cells if cell.cell_type == "code"]


def strip_trusted_flags(nb):
    """移除代码单元格的 trusted 标记（与服务端签名前的处理一致）。"""
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.metadata.pop("trusted", None)
    return nb


def record_for(notary, nb, generation=None):
    digest = notary.compute_signature(nb, generation=generation)
    return notary.store.get_record(digest, notary.algorithm)


def revoke_nb(notary, nb, reason=""):
    """按磁盘上的内容（无 trusted 标记）撤销笔记本签名。"""
    return notary.revoke_signature(strip_trusted_flags(nb), reason=reason)


# ---------------------------------------------------------------------------
# 签名记录内容
# ---------------------------------------------------------------------------


def test_signature_record_contains_generation_issued_at_and_normalization(notary, clock):
    nb = make_nb()
    notary.sign(nb)

    record = record_for(notary, nb)
    assert record is not None
    assert record.key_generation == 1
    assert record.issued_at == clock.now
    assert record.normalization_version == "v1"

    # 笔记本 metadata 中带有同样的提示信息
    hint = nb.metadata["signature_record"]
    assert hint["key_generation"] == 1
    assert hint["issued_at"] == clock.now.isoformat()
    assert hint["normalization_version"] == "v1"


def test_signature_hint_does_not_change_digest(notary):
    nb = make_nb()
    before = notary.compute_signature(nb)
    notary.sign(nb)  # 写入 metadata.signature_record 提示
    assert nb.metadata.get("signature_record")
    assert notary.compute_signature(nb) == before


def test_trust_status_hides_secret_material(notary):
    status = notary.trust_status()
    assert status["current_generation"] == 1
    assert status["normalization_version"] == "v1"
    assert "v1" in status["registered_normalizations"]
    for generation in status["keyring"]["generations"]:
        assert "secret" not in generation
        assert generation["has_secret"] is True


# ---------------------------------------------------------------------------
# 旧库迁移
# ---------------------------------------------------------------------------


def test_legacy_database_and_secret_migration(tmp_path):
    data_dir = tmp_path / "legacy"
    data_dir.mkdir()
    legacy = NotebookNotary(data_dir=str(data_dir))
    nb = make_nb()
    legacy.sign(nb)
    legacy_secret = (data_dir / "notebook_secret").read_bytes()
    legacy_digest = legacy.compute_signature(nb)

    migrated = GenerationalNotebookNotary(data_dir=str(data_dir))

    # legacy 密钥导入为第 1 代
    assert migrated.current_generation == 1
    assert migrated.keyring.secret_for(1) == legacy_secret
    # 旧记录归属到第 1 代且仍可验证
    record = migrated.store.get_record(legacy_digest, migrated.algorithm)
    assert record is not None
    assert record.key_generation == 1
    assert record.normalization_version == "v1"
    decision = migrated.evaluate_trust(nb)
    assert decision.trusted, decision.reason


def test_legacy_database_schema_gains_new_columns(tmp_path):
    data_dir = tmp_path / "legacy"
    data_dir.mkdir()
    legacy = NotebookNotary(data_dir=str(data_dir))
    legacy.sign(make_nb())
    db_file = legacy.db_file

    GenerationalNotebookNotary(data_dir=str(data_dir))

    db = sqlite3.connect(db_file)
    columns = {row[1] for row in db.execute("PRAGMA table_info(nbsignatures)")}
    db.close()
    for column in ("key_generation", "issued_at", "normalization_version", "revoked_at"):
        assert column in columns


def test_legacy_records_fail_closed_with_revocation_point(tmp_path, clock):
    """旧记录没有签发时间：设置撤销时点后无法证明产生于泄露之前，按不可信处理。"""
    data_dir = tmp_path / "legacy"
    data_dir.mkdir()
    legacy = NotebookNotary(data_dir=str(data_dir))
    nb = make_nb()
    legacy.sign(nb)

    migrated = GenerationalNotebookNotary(data_dir=str(data_dir))
    migrated._now = clock
    # 未设置撤销时点时保持可信
    assert migrated.evaluate_trust(nb).trusted

    migrated.rotate_key(
        verify_old_until=clock.now + timedelta(days=1),
        revoke_before=clock.now - timedelta(hours=1),
        reason="key leaked",
    )
    decision = migrated.evaluate_trust(nb)
    assert not decision.trusted
    assert decision.reason == UNKNOWN_ISSUANCE_TIME


# ---------------------------------------------------------------------------
# 轮换：验证窗口与撤销时点
# ---------------------------------------------------------------------------


def test_rotation_window_and_revocation_point(notary, clock):
    nb_old = make_nb("old = 1")
    notary.sign(nb_old)  # T0 签发，泄露前
    assert notary.evaluate_trust(nb_old).trusted

    # T0+2h 发现密钥在 T0+1h 泄露：轮换，撤销时点 T0+1h，窗口到 T0+1d
    clock.advance(hours=2)
    notary.rotate_key(
        verify_old_until=clock.now + timedelta(days=1),
        revoke_before=T0 + timedelta(hours=1),
        reason="key leaked",
    )
    assert notary.current_generation == 2

    # 泄露前签发的记录在窗口内仍可信
    decision = notary.evaluate_trust(nb_old)
    assert decision.trusted
    assert decision.key_generation == 1

    # 窗口关闭后不再可信
    expired = notary.evaluate_trust(nb_old, now=clock.now + timedelta(days=2))
    assert not expired.trusted
    assert expired.reason == VERIFICATION_WINDOW_EXPIRED

    # 泄露时点之后用旧密钥“签发”的记录不可信（伪造场景）
    forged = make_nb("forged = 1")
    from jupyter_server.services.contents.trust import SignatureRecord

    forged_digest = notary.compute_signature(forged, generation=1)
    notary.store.store_record(
        SignatureRecord(
            digest=forged_digest,
            algorithm=notary.algorithm,
            key_generation=1,
            issued_at=T0 + timedelta(hours=3),
            normalization_version="v1",
        )
    )
    forged_decision = notary.evaluate_trust(forged)
    assert not forged_decision.trusted
    assert forged_decision.reason == SIGNED_AFTER_REVOCATION_POINT

    # 新签名使用新代际
    nb_new = make_nb("new = 1")
    notary.sign(nb_new)
    record = record_for(notary, nb_new)
    assert record.key_generation == 2
    assert notary.evaluate_trust(nb_new).trusted


def test_rotation_without_revocation_point_keeps_old_signatures_valid(notary, clock):
    nb = make_nb()
    notary.sign(nb)
    notary.rotate_key(verify_old_until=clock.now + timedelta(days=1), reason="planned")
    assert notary.evaluate_trust(nb).trusted


def test_generation_revocation(notary, clock):
    nb1 = make_nb("one = 1")
    notary.sign(nb1)
    notary.rotate_key(reason="rotate")
    nb2 = make_nb("two = 2")
    notary.sign(nb2)

    notary.revoke_generation(1, reason="bulk compromise")
    d1 = notary.evaluate_trust(nb1)
    assert not d1.trusted
    assert d1.reason == KEY_GENERATION_REVOKED
    assert notary.evaluate_trust(nb2).trusted


def test_individual_revocation_is_sticky(notary):
    nb = make_nb()
    notary.sign(nb)
    assert notary.revoke_signature(nb, reason="bad content") == 1

    decision = notary.evaluate_trust(nb)
    assert not decision.trusted
    assert decision.reason == SIGNATURE_REVOKED

    # 同代际重签不会清除撤销标记
    notary.sign(nb)
    assert notary.evaluate_trust(nb).reason == SIGNATURE_REVOKED

    # 显式解除撤销后才能重新签名
    digest = notary.compute_signature(nb)
    assert notary.clear_revocation(digest) == 1
    notary.sign(nb)
    assert notary.evaluate_trust(nb).trusted


def test_unsign_removes_records_across_generations(notary, clock):
    nb = make_nb()
    notary.sign(nb)
    notary.rotate_key(reason="rotate")
    notary.sign(nb)  # 两代际都有记录
    notary.unsign(nb)
    decision = notary.evaluate_trust(nb)
    assert not decision.trusted
    assert decision.reason == UNSIGNED
    assert "signature_record" not in nb.metadata


# ---------------------------------------------------------------------------
# 信任继承重估：另存 / 复制 / 检查点恢复 / 外部修改
# ---------------------------------------------------------------------------


async def test_save_as_reevaluates_trust_inheritance(cm):
    nb = make_nb()
    await save_nb(cm, nb, "a.ipynb")
    assert trusted_flags(await get_nb(cm, "a.ipynb")) == [True]

    # 撤销后另存同一内容：不得继承信任
    revoke_nb(cm.notary, nb, reason="revoked before save-as")
    await save_nb(cm, nb, "b.ipynb")
    assert trusted_flags(await get_nb(cm, "b.ipynb")) == [False]


async def test_save_as_keeps_trust_when_record_valid(cm):
    nb = make_nb()
    await save_nb(cm, nb, "a.ipynb")
    await save_nb(cm, nb, "c.ipynb")
    assert trusted_flags(await get_nb(cm, "c.ipynb")) == [True]


async def test_cross_directory_copy_reevaluates_trust(cm, tmp_path):
    (tmp_path / "root" / "sub").mkdir()
    nb = make_nb()
    await save_nb(cm, nb, "a.ipynb")

    # 记录有效：跨目录复制继承信任
    await ensure_async(cm.copy("a.ipynb", "sub"))
    assert trusted_flags(await get_nb(cm, "sub/a.ipynb")) == [True]

    # 记录被撤销：复制不得继承信任
    revoke_nb(cm.notary, nb, reason="revoked before copy")
    (tmp_path / "root" / "sub2").mkdir()
    await ensure_async(cm.copy("a.ipynb", "sub2"))
    assert trusted_flags(await get_nb(cm, "sub2/a.ipynb")) == [False]


async def test_checkpoint_restore_reevaluates_trust(cm, caplog):
    nb = make_nb()
    await save_nb(cm, nb, "a.ipynb")
    checkpoints = await ensure_async(cm.list_checkpoints("a.ipynb"))
    assert checkpoints

    # 撤销记录后恢复检查点：信任继承必须重新判断
    revoke_nb(cm.notary, nb, reason="revoked before restore")
    with caplog.at_level("WARNING"):
        await ensure_async(cm.restore_checkpoint(checkpoints[0]["id"], "a.ipynb"))

    assert any("does not inherit trust" in r.message for r in caplog.records)
    assert trusted_flags(await get_nb(cm, "a.ipynb")) == [False]


async def test_external_modification_breaks_trust(cm, tmp_path):
    nb = make_nb()
    await save_nb(cm, nb, "a.ipynb")
    assert trusted_flags(await get_nb(cm, "a.ipynb")) == [True]

    # 外部直接修改文件内容
    os_path = tmp_path / "root" / "a.ipynb"
    data = json.loads(os_path.read_text())
    data["cells"][0]["source"] = "import os  # tampered"
    os_path.write_text(json.dumps(data))

    assert trusted_flags(await get_nb(cm, "a.ipynb")) == [False]


async def test_expired_window_blocks_automatic_resign_on_save(cm, clock):
    nb = make_nb()
    await save_nb(cm, nb, "a.ipynb")
    cm.notary.rotate_key(
        verify_old_until=clock.now + timedelta(hours=1), reason="planned rotation"
    )
    clock.advance(hours=2)  # 窗口已关闭

    # 窗口关闭后，同一内容再次保存不再自动继承信任（不会自动重签）
    nb2 = await get_nb(cm, "a.ipynb")
    nb2.cells[0].metadata.trusted = True  # 客户端声称可信
    await save_nb(cm, nb2, "a.ipynb")
    content = strip_trusted_flags(nb2)
    # 第 1 代记录仍在，且没有产生第 2 代记录
    assert record_for(cm.notary, content, generation=1) is not None
    assert record_for(cm.notary, content, generation=2) is None

    # 显式信任操作仍然可以重新签名
    await ensure_async(cm.trust_notebook("a.ipynb"))
    nb3 = await get_nb(cm, "a.ipynb")
    assert trusted_flags(nb3) == [True]
    assert record_for(cm.notary, strip_trusted_flags(nb3), generation=2).key_generation == 2


# ---------------------------------------------------------------------------
# 验证不执行输出内容
# ---------------------------------------------------------------------------


def test_verification_does_not_execute_outputs(notary):
    outputs = [
        new_output(
            "display_data",
            data={
                "text/html": "<script>alert(document.cookie)</script>",
                "application/javascript": "alert(1)",
            },
        ),
        new_output("execute_result", data={"text/html": "<b>rich</b>"}, execution_count=1),
    ]
    nb = make_nb(outputs=outputs)
    notary.sign(nb)
    snapshot = json.dumps(nb, sort_keys=True, default=str)

    def fail_if_executed(*args, **kwargs):  # pragma: no cover - 触即失败
        raise AssertionError("verification must not start kernels or execute outputs")

    with patch(
        "jupyter_client.KernelManager.start_kernel", side_effect=fail_if_executed
    ), patch("subprocess.Popen", side_effect=fail_if_executed):
        decision = notary.evaluate_trust(nb)

    assert decision.trusted
    # 笔记本内容（包括输出）在验证前后完全一致
    assert json.dumps(nb, sort_keys=True, default=str) == snapshot


# ---------------------------------------------------------------------------
# 批量重签任务
# ---------------------------------------------------------------------------


def test_resign_task_migrates_valid_and_skips_revoked(notary, clock):
    nbs = [make_nb(f"cell_{i} = {i}") for i in range(3)]
    for nb in nbs:
        notary.sign(nb)
    notary.revoke_signature(nbs[1], reason="revoked before migration")
    notary.rotate_key(reason="key leak")

    task = ResignTask(notary, task_id="migration")
    items = [(f"nb{i}.ipynb", nb) for i, nb in enumerate(nbs)]
    stats = task.run(items)

    assert stats["done"] is True
    assert stats["resigned"] == 2
    assert stats["skipped_revoked"] == 1

    # 有效签名迁移到新代际
    for nb in (nbs[0], nbs[2]):
        decision = notary.evaluate_trust(nb)
        assert decision.trusted
        assert decision.record.key_generation == 2
    # 已撤销签名保持撤销，且没有新代际记录
    assert notary.evaluate_trust(nbs[1]).reason == SIGNATURE_REVOKED
    assert record_for(notary, nbs[1], generation=2) is None


def test_resign_task_interrupt_and_resume(notary, clock):
    nbs = [make_nb(f"cell_{i} = {i}") for i in range(5)]
    for nb in nbs:
        notary.sign(nb)
    notary.rotate_key(reason="rotation")

    items = [(f"nb{i}.ipynb", nb) for i, nb in enumerate(nbs)]

    calls = {"n": 0}

    def stop_after_two():
        calls["n"] += 1
        return calls["n"] > 2

    first = ResignTask(notary, task_id="migration").run(items, should_stop=stop_after_two)
    assert first["done"] is False
    assert first["resigned"] == 2

    # 中断后重新运行：从断点继续，最终全部迁移
    second = ResignTask(notary, task_id="migration").run(items)
    assert second["done"] is True
    assert second["resigned"] == 3
    for nb in nbs:
        decision = notary.evaluate_trust(nb)
        assert decision.trusted
        assert decision.record.key_generation == 2


def test_resign_task_never_restores_revoked_after_window(notary, clock):
    nb = make_nb()
    notary.sign(nb)
    notary.rotate_key(
        verify_old_until=clock.now + timedelta(hours=1),
        revoke_before=clock.now,  # 撤销时点之前没有有效签名
        reason="leak",
    )
    stats = ResignTask(notary, task_id="t").run([("nb.ipynb", nb)])
    assert stats["resigned"] == 0
    assert not notary.evaluate_trust(nb).trusted


def test_iter_notebooks_skips_hidden_and_checkpoints(tmp_path):
    root = tmp_path / "root"
    (root / ".ipynb_checkpoints").mkdir(parents=True)
    (root / ".hidden").mkdir()
    (root / "sub").mkdir()
    import nbformat

    for rel in (
        "b.ipynb",
        "a.ipynb",
        "sub/c.ipynb",
        ".ipynb_checkpoints/b-checkpoint.ipynb",
        ".hidden/d.ipynb",
    ):
        nbformat.write(make_nb(), root / rel)

    keys = [key for key, _ in iter_notebooks(root)]
    assert keys == ["a.ipynb", "b.ipynb", "sub/c.ipynb"]


# ---------------------------------------------------------------------------
# HTTP 接口
# ---------------------------------------------------------------------------


async def test_trust_status_handler(jp_fetch):
    response = await jp_fetch("api", "trust", "status")
    assert response.code == 200
    status = json.loads(response.body)
    assert status["current_generation"] >= 1
    assert status["normalization_version"] == "v1"
    # 状态接口不暴露密钥材料
    for generation in status["keyring"]["generations"]:
        assert "secret" not in generation


async def test_trust_rotate_handler(jp_fetch):
    response = await jp_fetch(
        "api",
        "trust",
        "rotate",
        method="POST",
        body=json.dumps(
            {
                "verify_old_until": "2027-01-01T00:00:00+00:00",
                "revoke_before": "2026-01-01T00:00:00+00:00",
                "reason": "scheduled rotation",
            }
        ),
    )
    assert response.code == 201
    status = json.loads(response.body)
    assert status["current_generation"] == 2
    old = status["keyring"]["generations"][0]
    assert old["verify_until"] == "2027-01-01T00:00:00+00:00"
    assert old["revoke_before"] == "2026-01-01T00:00:00+00:00"


async def test_trust_rotate_handler_rejects_bad_datetime(jp_fetch):
    with pytest.raises(Exception) as exc_info:
        await jp_fetch(
            "api", "trust", "rotate", method="POST", body=json.dumps({"verify_old_until": "soon"})
        )
    assert "400" in str(exc_info.value)
