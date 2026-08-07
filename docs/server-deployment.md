# 实验室服务器部署指南

## 推荐目录

源码、原始数据和运行结果分开保存：

```text
/srv/cellvision/app          # 本项目源码
/srv/cellvision/raw          # 原始 TIFF 和 sessions.idx，只读挂载
/srv/cellvision/artifacts    # 候选、预测、审核结果和报告
/srv/cellvision/models       # 当前生产模型
/srv/cellvision/db           # SQLite 或 PostgreSQL 数据
/srv/cellvision/logs         # 服务和任务日志
```

Windows 服务器也应保持相同的逻辑分层，不要把原始图片复制进源码目录。

## 环境变量

复制 `.env.example` 后按服务器实际路径修改。配置文件中的相对路径仍可用于本地开发，服务器上优先使用：

- `CELLVISION_DATA_ROOT`
- `CELLVISION_ARTIFACT_ROOT`
- `CELLVISION_MODEL_ROOT`
- `CELLVISION_DB_ROOT`
- `CELLVISION_LOG_ROOT`
- `CELLVISION_HOST`
- `CELLVISION_PORT`

## 运行服务

本地或反向代理后端统一使用：

```powershell
.\.venv\Scripts\python.exe -m cellvision review-project `
  --manifest artifacts/projects/ql2603/project.json `
  --host 127.0.0.1 --port 8777
```

如果由容器或反向代理访问容器内部端口，需要明确设置 `CELLVISION_ALLOW_REMOTE=1`，不要默认暴露到实验室网络。

健康检查：

```text
GET /api/health
GET /api/ready
```

任务状态接口：

```text
GET  /api/project/tasks
GET  /api/project/tasks/{task_id}
POST /api/project/tasks/{task_id}/cancel
```

队列状态已经采用原子JSON写入并支持queued/running/completed/error/cancelled状态；实际项目推理executor应由独立GPU worker调用，不应放在HTTP请求线程中。

## 反向代理和权限

生产环境应在 Cell Vision 前放置 Nginx、Caddy 或 Traefik，并至少提供：

1. HTTPS；
2. 实验室账号认证；
3. 仅允许内网或VPN访问；
4. 请求体大小和超时限制；
5. `/api/health` 与 `/api/ready` 的独立探针。

当前应用本身还没有用户认证和角色权限，不能直接裸露到公网。

## 数据库与备份

单人审核阶段可继续使用每板SQLite；服务器多用户并发审核时建议迁移到 PostgreSQL。无论哪种方案，都要：

- 开启SQLite WAL和busy timeout；
- 对数据库、人工标注和最终报告做定期备份；
- 使用迁移版本号，不再只依赖运行时 `CREATE TABLE IF NOT EXISTS`；
- 备份前记录项目、板子、模型和配置版本。

## 任务执行

长时间推理和训练不能在HTTP请求中同步执行。生产目标是：

```text
浏览器 → API创建任务 → 持久化队列 → 单GPU worker → 阶段结果/进度 → 审核页面
```

单GPU服务器建议先配置一个GPU worker，避免多个板子同时抢占显存。

## CUDA

`deploy/Dockerfile`是便于验收的CPU基础镜像。正式GPU部署应选择与服务器驱动匹配的CUDA基础镜像，并安装对应的Torch/Torchvision版本；部署前执行：

```text
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

只有确认GPU可用后才启动推理worker，避免服务端误退回CPU导致任务耗时大幅增加。
