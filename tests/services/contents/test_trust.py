"""扩展信任管理的回归测试。

覆盖：签名记录字段（密钥代际、签发时间、内容规范化版本）、轮换时的
验证窗口与撤销时点、另存/恢复检查点/外部修改/跨目录复制后的信任继承
重判、验证过程不执行输出内容、可中断续跑的重签任务，以及已撤销签名
不得恢复为可信。
"""

import json
import time
from datetime import datetime, timedelta, timezone

import nbformat
import pytest
from jupyter_core.utils import ensure_async
from nbformat.sign import NotebookNotary
from nbformat.v4 import new_code_cell, new_notebook, new_output

from jupyter_server.services.contents.filemanager import (
    AsyncFileContentsManager,
    FileContentsManager,
)
from jupyter_server.services.contents.trust import (
    NORMALIZATION_VERSION,
    MemoryTrustStore,
    ResignInterrupted,
    TrustManager,
    TrustState,
)


def make_nb(source="print('hi')", unsafe=True):
    """构造一个（默认携带不信任输出类型的）笔记本。"""
    nb = new_notebook()
    outputs = []
    if unsafe:
        outputs = [new_output("display_data", {"application/javascript": "alert('hi');"})]
    nb.cells.append(new_code_cell(source, outputs=outputs))
    return nb


def assert_trusted(nb):
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert cell.metadata.get("trusted") is True


def assert_untrusted(nb):
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert not cell.metadata.get("trusted")


@pytest.fixture
def trust_dir(tmp_path):
    d = tmp_path / "trust"
    d.mkdir()
    return d


@pytest.fixture
def tm(trust_dir):
    manager = TrustManager(data_dir=str(trust_dir))
    yield manager
    manager.close()


@pytest.fixture(params=[FileContentsManager, AsyncFileContentsManager])
def cm(request, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    notary_dir = tmp_path / "notary"
    notary_dir.mkdir()
    trust_dir = tmp_path / "trust"
    trust_dir.mkdir()
    manager = request.param(root_dir=str(root))
    manager.notary = NotebookNotary(data_dir=str(notary_dir))
    trust_manager = TrustManager(data_dir=str(trust_dir))
    manager.trust_manager = trust_manager
    yield manager
    trust_manager.close()
    manager.notary.close()


async def save_trusted_nb(cm, path, source="print('hi')"):
    """保存一个笔记本并通过 trust_notebook 显式信任它。"""
    model = {"type": "notebook", "content": make_nb(source), "format": "json"}
    await ensure_async(cm.save(model, path))
    await ensure_async(cm.trust_notebook(path))


# ---------------------------------------------------------------------------
# 签名记录与验证
# ---------------------------------------------------------------------------


def test_signature_record_fields(tm):
    nb = make_nb()
    before = datetime.now(tz=timezone.utc)
    record = tm.sign_notebook(nb, "a.ipynb")
    after = datetime.now(tz=timezone.utc)
    # 签名记录必须包含密钥代际、签发时间与内容规范化版本
    assert record.key_generation == tm.current_generation
    assert before <= record.issued_at <= after
    assert record.issued_at.tzinfo is not None
    assert record.normalization_version == NORMALIZATION_VERSION
    assert record.algorithm == tm.algorithm
    assert record.status == "active"


def test_verify_unknown_and_trusted(tm):
    nb = make_nb()
    verdict = tm.verify_notebook(nb, "a.ipynb")
    assert verdict.state == TrustState.UNKNOWN
    assert not verdict.trusted
    tm.sign_notebook(nb, "a.ipynb")
    verdict = tm.verify_notebook(nb, "a.ipynb")
    assert verdict.state == TrustState.TRUSTED
    assert verdict.trusted


def test_verification_does_not_execute_outputs(tm, monkeypatch):
    nb = make_nb()
    tm.sign_notebook(nb, "a.ipynb")

    def boom(*args, **kwargs):
        raise AssertionError("验证过程不得执行笔记本的任何内容")

    monkeypatch.setattr("builtins.exec", boom)
    monkeypatch.setattr("builtins.eval", boom)
    verdict = tm.verify_notebook(nb, "a.ipynb")
    assert verdict.trusted


def test_verification_does_not_mutate_notebook(tm):
    nb = make_nb()
    tm.sign_notebook(nb, "a.ipynb")
    before = json.loads(json.dumps(nb))
    tm.verify_notebook(nb, "a.ipynb")
    after = json.loads(json.dumps(nb))
    assert before == after


def test_keyring_and_store_persist_across_restarts(trust_dir):
    tm1 = TrustManager(data_dir=str(trust_dir))
    nb = make_nb()
    tm1.sign_notebook(nb, "a.ipynb")
    tm1.rotate_key(verify_window=timedelta(hours=1))
    tm1.close()

    tm2 = TrustManager(data_dir=str(trust_dir))
    try:
        assert tm2.current_generation == 2
        verdict = tm2.verify_notebook(nb, "a.ipynb")
        assert verdict.trusted
        assert verdict.record.key_generation == 1
    finally:
        tm2.close()


def test_memory_store_backend(trust_dir):
    manager = TrustManager(data_dir=str(trust_dir), store_factory=lambda: MemoryTrustStore())
    nb = make_nb()
    manager.sign_notebook(nb, "a.ipynb")
    assert manager.verify_notebook(nb, "a.ipynb").trusted
    manager.revoke_notebook(nb)
    assert manager.verify_notebook(nb, "a.ipynb").state == TrustState.REVOKED
    manager.close()


# ---------------------------------------------------------------------------
# 密钥轮换：验证窗口与撤销时点
# ---------------------------------------------------------------------------


def test_rotation_keeps_old_generation_verifiable_within_window(tm):
    nb = make_nb()
    tm.sign_notebook(nb, "a.ipynb")
    tm.rotate_key(verify_window=timedelta(hours=1))
    # 验证窗口内，旧代际签名仍然可信
    verdict = tm.verify_notebook(nb, "a.ipynb")
    assert verdict.trusted
    assert verdict.record.key_generation == 1
    # 新签名使用新代际
    record = tm.sign_notebook(make_nb("y = 2"), "b.ipynb")
    assert record.key_generation == 2


def test_rotation_verification_window_expires(tm):
    nb = make_nb()
    tm.sign_notebook(nb, "a.ipynb")
    tm.rotate_key(verify_window=0)  # 验证窗口立即关闭
    time.sleep(0.01)
    verdict = tm.verify_notebook(nb, "a.ipynb")
    assert verdict.state == TrustState.EXPIRED
    assert not verdict.trusted


def test_rotation_revoked_before_distinguishes_pre_and_post_leak(tm):
    nb_before = make_nb("x = 1")
    tm.sign_notebook(nb_before, "before.ipynb")
    time.sleep(0.01)
    leak_time = datetime.now(tz=timezone.utc)
    time.sleep(0.01)
    nb_after = make_nb("x = 2")
    tm.sign_notebook(nb_after, "after.ipynb")

    tm.rotate_key(verify_window=timedelta(hours=1), revoked_before=leak_time)

    # 泄露之前签发的签名在窗口内仍然可信
    verdict_before = tm.verify_notebook(nb_before, "before.ipynb")
    assert verdict_before.trusted
    # 撤销时点及之后签发的签名视为泄露后伪造，不可信
    verdict_after = tm.verify_notebook(nb_after, "after.ipynb")
    assert verdict_after.state == TrustState.REVOKED
    assert not verdict_after.trusted


def test_revoke_generation(tm):
    nb = make_nb()
    tm.sign_notebook(nb, "a.ipynb")
    tm.revoke_generation(tm.current_generation)
    verdict = tm.verify_notebook(nb, "a.ipynb")
    assert verdict.state == TrustState.REVOKED
    assert not verdict.trusted


def test_stale_normalization_version(tm):
    nb = make_nb()
    record = tm.sign_notebook(nb, "a.ipynb")
    # 模拟规范化方案升级之前的旧记录
    record.normalization_version = 0
    state, reason = tm._evaluate_record(record, datetime.now(tz=timezone.utc))
    assert state == TrustState.STALE
    assert "规范化版本" in reason


# ---------------------------------------------------------------------------
# 撤销的签名不得恢复为可信
# ---------------------------------------------------------------------------


def test_revoked_signature_is_not_restored(tm):
    nb = make_nb()
    tm.sign_notebook(nb, "a.ipynb")
    assert tm.verify_notebook(nb, "a.ipynb").trusted

    assert tm.revoke_notebook(nb) == 1
    verdict = tm.verify_notebook(nb, "a.ipynb")
    assert verdict.state == TrustState.REVOKED
    assert not verdict.trusted

    # 拒绝为已撤销内容重新签发
    assert tm.sign_notebook(nb, "a.ipynb") is None
    # 轮换与重签任务同样不能复活已撤销的签名
    tm.rotate_key(verify_window=timedelta(hours=1))
    task = tm.begin_resign(["a.ipynb"])
    result = task.run(lambda path: nb)
    assert result.get("skipped-revoked") == 1
    assert tm.verify_notebook(nb, "a.ipynb").state == TrustState.REVOKED


async def test_revoked_notebook_stays_untrusted_through_manager(cm):
    await save_trusted_nb(cm, "a.ipynb")
    nb = (await ensure_async(cm.get("a.ipynb")))["content"]
    assert_trusted(nb)

    cm.trust_manager.revoke_notebook(nb)
    # 即使再次显式 trust，撤销也不会被覆盖
    await ensure_async(cm.trust_notebook("a.ipynb"))
    nb2 = (await ensure_async(cm.get("a.ipynb")))["content"]
    assert_untrusted(nb2)


# ---------------------------------------------------------------------------
# 另存、恢复检查点、外部修改、跨目录复制的信任继承重判
# ---------------------------------------------------------------------------


async def test_save_as_reevaluates_trust_inheritance(cm):
    await save_trusted_nb(cm, "a.ipynb")

    # 另存为新路径：信任继承被重新判断并延续
    model = await ensure_async(cm.get("a.ipynb"))
    await ensure_async(
        cm.save({"type": "notebook", "content": model["content"], "format": "json"}, "b.ipynb")
    )
    nb_b = (await ensure_async(cm.get("b.ipynb")))["content"]
    assert_trusted(nb_b)

    # 撤销之后另存不再继承信任
    cm.trust_manager.revoke_notebook(nb_b)
    await ensure_async(cm.save({"type": "notebook", "content": nb_b, "format": "json"}, "c.ipynb"))
    nb_c = (await ensure_async(cm.get("c.ipynb")))["content"]
    assert_untrusted(nb_c)


async def test_checkpoint_restore_reevaluates_trust(cm):
    await save_trusted_nb(cm, "a.ipynb")
    nb = (await ensure_async(cm.get("a.ipynb")))["content"]
    assert_trusted(nb)
    checkpoint = await ensure_async(cm.create_checkpoint("a.ipynb"))

    # 撤销签名后恢复检查点：已撤销的签名不得借此恢复为可信
    cm.trust_manager.revoke_notebook(nb)
    await ensure_async(cm.restore_checkpoint(checkpoint["id"], "a.ipynb"))
    nb2 = (await ensure_async(cm.get("a.ipynb")))["content"]
    assert_untrusted(nb2)


async def test_external_modification_reevaluates_trust(cm):
    await save_trusted_nb(cm, "a.ipynb")
    nb = (await ensure_async(cm.get("a.ipynb")))["content"]
    assert_trusted(nb)

    # 在内容管理器之外直接改写文件
    os_path = cm._get_os_path("a.ipynb")
    with open(os_path, encoding="utf-8") as f:
        disk_nb = nbformat.read(f, as_version=4)
    disk_nb.cells.append(new_code_cell("import os"))
    with open(os_path, "w", encoding="utf-8") as f:
        nbformat.write(disk_nb, f)

    nb2 = (await ensure_async(cm.get("a.ipynb")))["content"]
    assert_untrusted(nb2)


def test_external_modification_is_detected(tm):
    nb = make_nb()
    tm.sign_notebook(nb, "a.ipynb")
    nb.cells.append(new_code_cell("import os"))
    verdict = tm.verify_notebook(nb, "a.ipynb", event="read")
    assert not verdict.trusted
    assert verdict.external_change


async def test_cross_directory_copy_reevaluates_trust(cm):
    for d in ("src", "dst", "dst2"):
        await ensure_async(cm.new({"type": "directory"}, d))
    await save_trusted_nb(cm, "src/a.ipynb")

    # 跨目录复制：信任继承被重新判断并延续
    await ensure_async(cm.copy("src/a.ipynb", "dst"))
    nb = (await ensure_async(cm.get("dst/a.ipynb")))["content"]
    assert_trusted(nb)

    # 撤销之后跨目录复制不再继承信任
    cm.trust_manager.revoke_notebook(nb)
    await ensure_async(cm.copy("src/a.ipynb", "dst2"))
    nb2 = (await ensure_async(cm.get("dst2/a.ipynb")))["content"]
    assert_untrusted(nb2)


async def test_reevaluate_trust_entrypoint(cm):
    await save_trusted_nb(cm, "a.ipynb")
    verdict = await ensure_async(cm.reevaluate_trust("a.ipynb", event="save"))
    assert verdict.trusted
    # 非笔记本路径不参与信任重判
    assert await ensure_async(cm.reevaluate_trust("a.txt", event="save")) is None


# ---------------------------------------------------------------------------
# 可中断续跑的重签任务
# ---------------------------------------------------------------------------


def test_resign_migrates_to_current_generation(tm):
    nb = make_nb()
    tm.sign_notebook(nb, "a.ipynb")
    tm.rotate_key(verify_window=timedelta(hours=1))

    task = tm.begin_resign(["a.ipynb"])
    result = task.run(lambda path: nb)
    assert result.get("resigned") == 1

    verdict = tm.verify_notebook(nb, "a.ipynb")
    assert verdict.trusted
    assert verdict.record.key_generation == tm.current_generation

    # 已经是最新代际的内容不会重复重签
    task2 = tm.begin_resign(["a.ipynb"])
    assert task2.run(lambda path: nb).get("skipped-current") == 1


def test_resign_interrupt_and_resume(tm):
    nbs = {}
    paths = []
    for i in range(3):
        nb = make_nb(f"x = {i}")
        path = f"nb{i}.ipynb"
        tm.sign_notebook(nb, path)
        nbs[path] = nb
        paths.append(path)
    # 第三个笔记本的签名被撤销
    tm.revoke_notebook(nbs[paths[2]])
    tm.rotate_key(verify_window=timedelta(hours=1))

    task = tm.begin_resign(paths)
    processed = []

    def loader(path):
        processed.append(path)
        if len(processed) == 2:
            task.interrupt()
        return nbs[path]

    with pytest.raises(ResignInterrupted):
        task.run(loader)
    status = task.status()
    assert status.get("resigned") == 2
    assert status.get("pending") == 1

    # 中断后可以继续，已完成的条目不会重复处理
    task2 = tm.resume_resign(task.task_id)
    result = task2.run(lambda path: nbs[path])
    assert "pending" not in result
    assert result.get("resigned") == 2
    assert result.get("skipped-revoked") == 1

    # 前两个笔记本已迁移到新代际
    verdict = tm.verify_notebook(nbs[paths[0]], paths[0])
    assert verdict.trusted
    assert verdict.record.key_generation == tm.current_generation
    # 已撤销的签名不会被重签任务恢复为可信
    verdict2 = tm.verify_notebook(nbs[paths[2]], paths[2])
    assert verdict2.state == TrustState.REVOKED
    assert not verdict2.trusted


def test_resign_resume_after_restart(trust_dir):
    tm1 = TrustManager(data_dir=str(trust_dir))
    nb = make_nb()
    tm1.sign_notebook(nb, "a.ipynb")
    tm1.rotate_key(verify_window=timedelta(hours=1))
    task = tm1.begin_resign(["a.ipynb"])
    task_id = task.task_id
    tm1.close()

    # 模拟进程重启后继续任务
    tm2 = TrustManager(data_dir=str(trust_dir))
    try:
        task2 = tm2.resume_resign(task_id)
        result = task2.run(lambda path: nb)
        assert result.get("resigned") == 1
        assert tm2.verify_notebook(nb, "a.ipynb").record.key_generation == 2
    finally:
        tm2.close()


def test_resume_unknown_resign_task(tm):
    with pytest.raises(KeyError):
        tm.resume_resign("no-such-task")
