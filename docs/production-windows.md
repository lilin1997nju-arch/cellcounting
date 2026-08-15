# Windows 生产推理环境

生产端只做识别计算，不做模型训练。建议把生产机拆成四类目录：

```text
cell-vision/              # 程序代码和启动脚本
cellvision-data/         # 原始导出数据，只读使用
cellvision-models/       # 发布模型，只读使用
cellvision-artifacts/    # 推理结果、项目状态和缓存
cellvision-db/           # 预留的数据库目录；当前版本的 annotations.db 随项目 artifact 保存
cellvision-logs/         # 服务和 worker 日志
```

## 需要准备的内容

- Windows 10/11 x64 或 Windows Server。
- Python 3.11--3.13；脚本优先使用 `py -3.12`。
- 原始数据目录，包含项目导入所需的 `sessions.idx`、TIFF/CF 图像和仪器导出的 cells CSV。
- 一个项目清单 `project.json`。它可以由项目服务创建，也可以从开发环境复制到生产机；清单中的原始数据路径必须是生产机可访问的绝对路径。
- 发布模型包中的四个文件：

```text
<MODEL_ROOT>/models/teaching_classifier.pt
<MODEL_ROOT>/models/multiplicity_classifier.pt
<MODEL_ROOT>/v2/models/latest_instance_segmenter.pt
<MODEL_ROOT>/v2/models/latest_temporal_evidence.pt
```

GPU 不是必需条件。GPU 机器需要可用的 NVIDIA 驱动；不要求单独安装 CUDA Toolkit。脚本先检查 `nvidia-smi`/NVIDIA 适配器，再安装官方 PyTorch CUDA wheel，最后以 `torch.cuda.is_available()` 的真实结果决定是否使用 CUDA。任何驱动或 CUDA 探测失败都会自动回退到 CPU。

## 一键配置

先把整个项目目录复制到生产机，例如 `E:\CellVision`，再在该目录打开 PowerShell：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_production.ps1 `
  -InstallRoot E:\CellVision `
  -DataRoot E:\CellVisionData `
  -ArtifactRoot E:\CellVisionArtifacts `
  -ModelRoot E:\CellVisionModels `
  -DbRoot E:\CellVisionDb `
  -LogRoot E:\CellVisionLogs `
  -Manifest E:\CellVisionArtifacts\projects\active\project.json `
  -Device auto
```

如果模型暂时还未复制到 `ModelRoot`，可先加 `-SkipModelCheck` 完成 Python 环境配置；正式启动前仍必须补齐模型文件。

安装完成后，启动服务和计算 worker：

```powershell
.\scripts\start_production.ps1 -InstallRoot E:\CellVision
```

默认监听 `127.0.0.1:8777`，服务会隐藏运行，并把日志写入 `CELLVISION_LOG_ROOT`。诊断时可使用：

```powershell
.\scripts\start_production.ps1 -InstallRoot E:\CellVision -Foreground
```

启动前脚本会检查模型包、端口占用和运行时；启动后会等待：

```text
http://127.0.0.1:8777/api/health
http://127.0.0.1:8777/api/ready
```

重启同一个 Cell Vision 服务：

```powershell
.\scripts\start_production.ps1 -InstallRoot E:\CellVision -Restart
```

## CPU/GPU 判断规则

```text
Device=auto
  ├─ NVIDIA 驱动/适配器不存在或不可用 → 安装 CPU PyTorch → CPU worker
  ├─ CUDA wheel 可安装且 torch.cuda.is_available()=True → CUDA worker
  └─ CUDA wheel/驱动探测失败 → 安装 CPU PyTorch → CPU worker
```

也可以强制指定：

```powershell
# 即使存在 GPU 也只用 CPU
.\scripts\setup_production.ps1 -Device cpu

# 偏好 CUDA；CUDA 真正不可用时仍安全回退 CPU
.\scripts\setup_production.ps1 -Device cuda
```

运行时详情：

```powershell
& E:\CellVision\.venv-production\Scripts\python.exe -m cellvision runtime-info --device auto
```

## 生产模式边界

脚本生成的 `.env.production` 会设置 `CELLVISION_PRODUCTION=1`。此模式下：

- `cellvision train ...` 会直接拒绝执行；
- `/api/teach-train`、`/api/integrated-review-new-round`、`/api/auto-review-new-round` 返回 403；
- worker 只加载已发布 checkpoint 做推理，并把结果写入 artifact/DB/log 目录；
- 原始图像和模型目录应赋予只读权限；
- `generate_auto_annotation_round`、`generate_integrated_training_round` 等名称中的 “round” 是推理结果/审核队列生成，不会更新模型 checkpoint，不等同于训练。
- 当前版本的 `CELLVISION_DB_ROOT` 仅作为目录约定保留；实际每个项目的 `annotations.db` 位于该项目的 artifact 目录中，不能只备份 `cellvision-db`。

训练请在开发环境完成，验收后只把模型包和必要的配置/项目清单发布到生产机。

## 网络和数据安全

默认只绑定回环地址。如果要让其他电脑访问，应使用内网/VPN 加反向代理，并明确传入 `-AllowRemote`；应用本身没有完整的用户认证，不应直接暴露到公网。原始图像建议放在只读磁盘或只读网络共享上，artifact、DB、log 放在可写磁盘并纳入备份。

PyTorch 的 Windows 安装方式和 CPU/CUDA wheel 选择以官方安装页为准；脚本固定使用官方 wheel 索引，并在安装后再次验证实际 CUDA 能力。
