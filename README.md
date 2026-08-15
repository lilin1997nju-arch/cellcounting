# Cell Vision

面向 96 孔板多时间点显微图像的细胞/杂质识别、审核和孔级生长筛选系统。

当前主流程：

1. 解析 `sessions.idx`，建立板子、孔位和实际 Day 时间点 Manifest；
2. 使用末点图像快速判断孔是否有明显生长；
3. 对阳性孔执行 T0/T1/T2 候选生成、实例分割、形态/多重性和时序判定；
4. 输出孔级报告并在项目级审核页面复核；
5. 人工审核结果写入数据库，作为后续训练数据。

详细架构见 [`docs/architecture.md`](docs/architecture.md)，服务器部署见 [`docs/server-deployment.md`](docs/server-deployment.md)。

## 本地安装

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

建议复制 `.env.example` 为 `.env`，把原始数据、模型、数据库和 artifact 分开配置。

## Windows 生产推理部署

生产端只需要计算，不需要训练模型。使用 [`docs/production-windows.md`](docs/production-windows.md) 中的 `scripts/setup_production.ps1` 一键创建独立的 `.venv-production`；脚本会自动判断 NVIDIA/CUDA 是否可用，GPU 不可用时安装 CPU PyTorch 并回退到 CPU worker。模型发布包必须包含 `models/` 和 `v2/models/` 下的四个 checkpoint，生产模式会禁止训练 CLI 和训练接口。

## 常用命令

```powershell
# 解析 sessions.idx
.\.venv\Scripts\python.exe -m cellvision parse-sessions `
  --root E:\CM\20260623\QL2603 `
  --output artifacts\ingest\sessions.csv

# 构建单板 Manifest
.\.venv\Scripts\python.exe -m cellvision build-manifest --config configs/default.yaml

# 启动项目审核服务（固定使用 8777，避免重复开端口）
.\.venv\Scripts\python.exe -m cellvision review-project `
  --manifest artifacts/projects/ql2603/project.json `
  --host 127.0.0.1 --port 8777
```

Windows 下如果不希望服务占用前台 PowerShell 窗口，可使用后台模式：

```powershell
.\.venv\Scripts\python.exe -m cellvision review-project `
  --manifest artifacts/projects/ql2603/project.json `
  --host 127.0.0.1 --port 8777 --background
```

后台服务默认关闭逐请求访问日志，日志写入 `artifacts/logs/review-project-8777.stdout.log` 和 `.stderr.log`。

健康检查：

```text
http://127.0.0.1:8777/api/health
http://127.0.0.1:8777/api/ready
```

项目目录使用 `artifacts/projects/project_catalog.sqlite` 作为可重建的项目索引。主页的搜索和分页由 SQL 服务端执行；任务、板子、孔当前结论和审核历史会在对应保存链路完成后同步。手动重建/校验：

```powershell
.\.venv\Scripts\python.exe -m cellvision catalog-sync `
  --manifest artifacts/projects/ql2603/project.json --force
```

状态接口：`GET /api/catalog/status`；单板孔结论接口：`GET /api/project/catalog-wells?project_id=...&plate_slug=...`。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## 数据和结果

原始图像只读。训练数据、模型、预测、缓存、报告和审核数据库均写入 `artifacts/` 或服务器配置的外部目录。不要把 `.venv`、缓存、原始图像和大模型文件提交到Git；归档策略见 [`docs/artifact-retention.md`](docs/artifact-retention.md)。

## 当前限制

- 任务队列目前已能保存任务计划，但正式的后台 worker、重试和取消机制仍需继续补齐；
- 当前审核服务尚未提供用户认证，生产部署必须放在反向代理和实验室内网/VPN后面；
- 多用户并发审核时建议从按板 SQLite 迁移到 PostgreSQL。
## 自适应计算 worker

项目审核服务默认会启动一个后台 worker，启动时自动检测 CUDA；有 GPU 使用 GPU，没有 GPU 自动使用 CPU。也可以单独运行：

```powershell
.\.venv\Scripts\python.exe -m cellvision project-worker `
  --manifest artifacts\projects\ql2603\project.json `
  --device auto
```

部署说明见 [`docs/worker-deployment.md`](docs/worker-deployment.md)。
