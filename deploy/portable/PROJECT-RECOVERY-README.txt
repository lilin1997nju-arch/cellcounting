Cell Vision 项目列表恢复工具
============================

适用情况
--------
Workspace\Projects 中的项目文件仍在，但 Cell Vision 主页看不到部分或全部项目。

使用方法
--------
1. 关闭正在浏览的 Cell Vision 页面即可，不需要停止服务。
2. 将恢复包内的全部文件直接复制到 Cell Vision 安装目录根目录。
   正确位置应与 Application、Workspace、Configure-CellVision-Service.cmd 同级。
3. 双击 Recover-CellVision-Projects.cmd。
4. 等待窗口显示恢复完成，然后刷新 Cell Vision 主页。

安全行为
--------
- 执行前会把现有 project.json、任务元数据和项目目录数据库备份到：
  Workspace\Projects\.metadata-backups\<时间戳>\
- 不会删除 Workspace 中的任何项目、图片、报告或数据库文件。
- 优先从有效备份原样恢复 project.json。
- 没有备份时，只有在目录数据库记录对应的全部板子产物均完整时才会重建 project.json。
- 不完整项目会显示 Skipped 和具体原因，原文件保持不变。
- 服务已运行时不会重启或重装；服务停止时会请求管理员权限启动。
- 如果 Windows 服务实际属于另一份安装目录，工具会停止并显示两个目录，避免刷新错误的 Workspace。

日志位置
--------
Workspace\Logs\cellvision-project-recovery.log
Workspace\Logs\cellvision-daily-launch.log

如果仍有项目显示 Skipped，请将上述日志和对应项目文件夹结构发给开发人员分析。
