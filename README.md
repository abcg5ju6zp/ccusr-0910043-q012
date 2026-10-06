# Jupyter Server 内容服务

本项目提供服务端内容、目录、检查点、会话和鉴权接口。生产源码位于 `jupyter_server/`，核心回归测试位于 `tests/`。

## 安装

`python3 -m pip install --break-system-packages --no-build-isolation -e '.[test]'`

## 测试

`python3 -m pytest -q`

## 构建

`python3 -m compileall -q jupyter_server`

`python3 -m build --wheel --no-isolation`

## 使用

内容管理器可在本地目录上执行保存、复制、改名、删除和检查点操作，HTTP 处理器提供对应服务端接口。

## 信任管理

`jupyter_server/services/contents/trust.py` 在 nbformat 的 `NotebookNotary` 之上提供面向密钥轮换的扩展信任管理：

- 每条签名记录携带密钥代际、签发时间与内容规范化版本；
- `TrustManager.rotate_key(verify_window=..., revoked_before=...)` 轮换密钥时可设置旧代际的验证窗口与撤销时点，安全人员据此区分签名产生于密钥泄露之前还是之后；
- 另存、恢复检查点、外部修改与跨目录复制之后都会重新判断信任继承（`ContentsManager.reevaluate_trust`）；
- 验证只对规范化后的内容做哈希比对，不执行笔记本的代码或输出内容；
- `TrustManager.begin_resign(...)` / `resume_resign(task_id)` 提供可中断续跑的重签任务，进度逐条持久化，已撤销的签名不会被恢复为可信。

对应回归测试位于 `tests/services/contents/test_trust.py`。

