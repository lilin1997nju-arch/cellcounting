# 自适应 GPU/CPU Worker

项目 worker 启动时会调用 PyTorch 检测 CUDA：

- 检测到可用 CUDA 时，自动选择 GPU worker，并记录 GPU 名称、CUDA 版本和 PyTorch 版本。
- 没有 GPU、驱动不可用或 CUDA 检测失败时，自动切换 CPU worker，不会阻止任务运行。
- `queued` 仍表示“已加入队列但尚未开始”；点击“开始计算”后，worker 才会接管任务并更新板级进度。

## 与审核服务一起启动

`review-project` 默认会在同一进程中启动一个后台自适应 worker：

```powershell
.\.venv\Scripts\python.exe -m cellvision review-project `
  --manifest artifacts\projects\ql2603\project.json `
  --host 127.0.0.1 --port 8777
```

启动日志会输出 `worker_started`，任务进度也会显示实际使用的 GPU 或 CPU。若只想运行审核服务，可加 `--no-worker`。

## 独立 worker 进程

需要把计算放到另一台电脑时，在那台电脑部署相同代码、模型和可访问的数据目录，然后运行：

```powershell
.\.venv\Scripts\python.exe -m cellvision project-worker `
  --manifest artifacts\projects\ql2603\project.json `
  --device auto
```

独立 worker 与 8777 服务共享 `task_queue.json`。同一份队列只运行一个 worker，避免多个进程同时抢占同一块 GPU。

可选覆盖：

```powershell
# 明确使用 CPU，即使本机存在 GPU
.\.venv\Scripts\python.exe -m cellvision project-worker --manifest ... --device cpu

# 明确偏好 CUDA；如果 CUDA 实际不可用，仍会安全回退 CPU
.\.venv\Scripts\python.exe -m cellvision project-worker --manifest ... --device cuda
```

## 状态检查

```text
GET /api/project/worker-runtime
GET /api/project/tasks
```

worker 状态保存在队列目录的 `worker_runtime.json`，其中包含 `selected_device`、GPU 名称、CUDA 版本、进程号和当前状态。取消运行中的任务后，worker 会终止当前板子的子进程，并把任务标记为 `cancelled`；原始数据不会被删除。

