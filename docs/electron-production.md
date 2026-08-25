# Cell Vision Electron 生产部署

## 交付结构

`CellVision-Setup-0.2.1-x64.exe` 是面向 Windows 10/11 x64 的单文件离线安装程序，包含：

- Electron 43.4.1 桌面客户端；
- CPU 版 Python 3.12、PyTorch、模型及生产服务；
- Microsoft Visual C++ x64 运行库安装程序；
- 项目元数据备份/恢复和历史 Workspace 导入工具。

安装程序只需要首次运行时确认管理员权限。安装完成后，日常只启动桌面快捷方式 `Cell Vision`。
安装程序会同时创建桌面快捷方式和开始菜单快捷方式；按当前产品选择保留 Electron 默认原子轨道图标。

## Workspace 保护

- 安装和升级只替换客户端与 `Application`，不将 Workspace 打入安装包；
- NSIS 卸载/升级删除逻辑明确排除 `$INSTDIR\Workspace`；
- 服务配置、项目修复和 Workspace 导入前都会备份项目元数据；
- 卸载客户端后 Workspace 仍保留，需管理员明确手工删除才会移除历史数据。

安装版通过 `Workspace\Projects\<项目目录>\project.json` 识别项目，并在服务启动或强制刷新目录时同步到项目列表。只有原始图片或结果子目录、但缺少有效 `project.json` 的目录不会直接显示；“修复/重新扫描项目目录”会先尝试从元数据备份恢复，或在计算产物完整时安全重建清单。

如果在相同目录覆盖安装，已有 Workspace 会被自动保留并重新索引，无需复制数据。如果旧 Workspace 位于其他目录，可在客户端“维护 → 导入历史 Workspace”中选择旧的 `Workspace` 文件夹，也可以选择它的上一级安装目录。导入时会：

1. 临时停止生产服务；
2. 备份当前项目元数据；
3. 复制不存在的项目、Inbox 和 Database 文件，不覆盖同名项目或文件；
4. 将新复制 JSON 元数据中的旧 Workspace 绝对路径转换为新路径；
5. 启动服务并强制刷新项目目录。

也可以在首次启动前将完整旧 Workspace 放到新安装目录根目录；启动后服务会扫描其中的项目。如果是在客户端运行后手工复制，则使用“维护 → 修复/重新扫描项目目录”强制刷新。

导入不会合并同名项目目录：目标 Workspace 已存在同名项目时会跳过，避免覆盖当前项目。如果确实需要替换或合并同名项目，应先单独备份并人工确认处理范围。

## 锁屏和后台计算

`CellVisionProduction` 以 `LocalSystem` 身份运行在 Windows Session 0，CPU计算工作器也在 Session 0，不依赖 Electron窗口或登录桌面。因此锁屏、关闭客户端窗口或退出当前登录桌面不会停止已经开始的计算。

锁屏不等于睡眠：机器进入睡眠或休眠后CPU不再运行，计算会暂停到系统恢复。生产电脑应允许关闭显示器，但禁用自动睡眠和休眠。

## 不同时间点板子交集

任务创建会对 Day0、Day1、Day2和所选末点取完整板子交集：

- 缺少任一所选时间点的板子自动排除；
- 任一所选时间点96孔文件不完整的板子自动排除；
- 任务计划保存纳入板子、排除板子及逐板原因；
- 计算工作器再次按任务计划中的交集过滤，排除板不进入计算；
- 只有交集为0时才阻止任务创建并提示。

## 构建

从已有CPU便携生产运行时构建：

```powershell
scripts\build_electron_installer.ps1 `
  -PortableReleaseRoot release\CellVision-offline-6231fdce81-cpu
```

构建脚本会覆盖最新源码和审核界面，下载并校验微软签名的 VC++ x64 运行库，然后生成 `electron\dist\CellVision-Setup-0.2.1-x64.exe`。

当前内部交付程序没有商业代码签名证书，Windows可能显示“未知发布者”或 SmartScreen 提示；这不影响离线运行，但正式大规模发布前应使用组织代码签名证书签署安装程序。
