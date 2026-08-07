# 审核界面说明

当前生产入口是项目级审核服务的 `/`：

- `/`：项目列表；
- `/projects/{project_id}/`：项目下板子和孔级结论；
- `/plates/{plate_slug}/auto-review`：T0/T1/T2按孔快速审核；
- `/plates/{plate_slug}/teach`、`/doublet-teach`、`/integrated-review`：训练和管理员页面。

旧版 `screening.*` 和根目录 `index.html/app.js/styles.css` 仍暂时保留，用于历史结果兼容。迁移完成并确认没有引用后再归档。

