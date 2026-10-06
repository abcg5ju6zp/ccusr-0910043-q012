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

## 分代际信任管理

`jupyter_server.services.contents.trust` 在 nbformat 签名模型之上提供：

- 签名记录包含**密钥代际**、**签发时间**和**内容规范化版本**，笔记本 metadata 中带有同样的提示（不参与摘要）。
- `GenerationalNotebookNotary.rotate_key(verify_old_until=..., revoke_before=...)` 轮换密钥：撤销时点之前签发的记录在验证窗口内仍可验证；撤销时点之后的一律不可信。无签发时间的旧记录在设置撤销时点后按不可信处理（fail-closed）。
- 另存、恢复检查点、外部修改、跨目录复制时自动重新判断信任继承；保存路径不会把已撤销的签名恢复为可信（显式 `trust_notebook` 除外）。
- 验证只对内容做 HMAC，不执行输出内容。
- `ResignTask` 批量重签：进度持久化，可中断续跑，跳过已撤销签名。

HTTP 接口：`GET /api/trust/status` 查看密钥代际与轮换策略（不含密钥材料）；`POST /api/trust/rotate` 携带 `verify_old_until` / `revoke_before` / `reason` 执行轮换。
