Cell Vision 桌面版 ZIP 安装说明
================================

一、安装
--------
1. 将整个 ZIP 复制到生产电脑本机硬盘，不要从共享盘或 ZIP 预览窗口直接运行文件。
2. 使用“全部解压”把 ZIP 直接解压到最终安装目录，例如 D:\CellVision。
3. 建议最终路径尽量简短，完整安装路径不要超过约90个字符。
4. 解压完成后，确认 Cell Vision.exe、Application、Install-CellVision.cmd 位于同一目录。
5. 双击 Install-CellVision.cmd，并允许一次管理员权限。

安装入口会继续完成以下操作：
- 安装或更新随包提供的微软 VC++ x64 运行库；
- 使用包内自带的全部计算模型（包含 ResNet18 特征权重），计算过程不联网下载模型；
- 创建并保留 Workspace；
- 配置独立的 CellVisionDesktopProduction Windows 服务；
- 如果8777或当前配置端口已被旧服务占用，自动选择后续空闲端口并保存；
- 核对 Workspace 实例标识，防止客户端误连其他服务；
- 创建桌面和开始菜单快捷方式；
- 配置完成后启动 Cell Vision 客户端。

二、历史项目与覆盖安装
----------------------
- ZIP 不包含 Workspace，解压到同一安装目录不会主动删除历史项目。
- 覆盖更新前必须先退出 Cell Vision 客户端，并停止本安装目录对应的桌面服务，避免运行中的
  Python/DLL 文件无法被替换。也可以解压到一个新目录，再通过“维护 → 导入历史 Workspace”迁移。
- 不要删除旧 Workspace。手工复制时应复制完整 Workspace，而不是只复制 Projects 子目录。

三、日志与故障排查
------------------
安装日志：Workspace\Logs\cellvision-zip-install-*.log
日常启动日志：Workspace\Logs\cellvision-daily-launch.log

如果提示 ZIP 未完整解压，请不要在压缩包预览窗口中双击安装入口，必须先执行“全部解压”。
如果提示模型文件缺失或校验失败，说明 ZIP 未完整解压或文件被安全软件隔离，请重新完整解压；
不要通过临时关闭证书校验来绕过该错误。
