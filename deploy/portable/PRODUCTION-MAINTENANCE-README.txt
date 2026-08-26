Cell Vision 生产端启动与项目目录恢复更新包
============================================

一、安装方法
------------
1. 解压本压缩包。
2. 将压缩包内的全部文件直接复制到 Cell Vision 安装目录根目录，并覆盖同名脚本。
3. 正确位置应与 Application、Workspace 文件夹同级。
4. 本更新包不包含 Application、Workspace、模型或项目数据，不要移动或删除这些目录。

二、恢复现有项目列表
--------------------
1. 双击 Recover-CellVision-Projects.cmd。
2. 等待窗口显示恢复完成。
3. 刷新 Cell Vision 主页。

恢复前会自动备份项目元数据到：
Workspace\Projects\.metadata-backups\<时间戳>\

恢复过程不会删除任何项目文件。无法安全恢复的残缺项目会显示 Skipped 和具体原因，
日志保存在 Workspace\Logs\cellvision-project-recovery.log。

三、以后日常启动
----------------
每天双击 Start-CellVision.cmd。

服务已经运行时不会重启或重装；服务停止时只启动服务，并检查 API 和计算工作器，
随后打开平台。日志保存在 Workspace\Logs\cellvision-daily-launch.log。

桌面版使用独立的 CellVisionDesktopProduction 服务。若 8777 已被旧版服务或其他程序占用，
启动器会自动选择后续空闲端口并写入 Application\.env.production；端口切换时只需确认一次
管理员授权。客户端还会核对 Workspace 实例标识，不会误连旧服务或打开错误的项目目录。

四、服务配置入口
----------------
Configure-CellVision-Service.cmd 只用于首次安装服务或服务确实损坏时重新配置，
不要作为日常启动入口。

更新后的配置脚本会先备份项目元数据；同一安装目录下不再卸载重装桌面服务；
从旧桌面包升级时，会将本安装目录原有的 CellVisionProduction 迁移为独立桌面服务名；
其他安装目录中的旧服务不会被停止或删除。
如果检测到服务属于另一份 Cell Vision 安装目录，会默认停止并提示，避免切换到错误的 Workspace。

五、文件用途
------------
Start-CellVision.cmd                 日常一键启动入口
start_cellvision_platform.ps1        日常启动执行逻辑
Recover-CellVision-Projects.cmd      项目列表恢复入口
recover_cellvision_projects.ps1      项目列表恢复执行逻辑
repair_project_manifests.py          项目元数据备份与安全恢复
Configure-CellVision-Service.cmd     服务配置入口（非日常使用）
configure_service_launcher.ps1       服务配置管理员启动器
configure_service.ps1                安全服务配置逻辑
PROJECT-RECOVERY-README.txt           项目恢复详细说明
