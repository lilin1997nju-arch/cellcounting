# Artifact 保留和归档策略

## 必须长期保留

- 原始数据和 `sessions.idx`；
- 人工审核数据库；
- 当前生产模型；
- 训练集和验证集标注快照；
- 最终孔级报告；
- 每次正式运行的配置、模型版本和输入指纹。

## 可重新生成、可按周期清理

- `artifacts/cache/` 下的图像和候选缓存；
- 预览 JPEG/PNG；
- `teaching_features.npz` 等特征缓存；
- 已完成且已归档的中间 CSV；
- `.pytest_cache`、`__pycache__`、`.pytest-tmp`、`src/cellvision_local.egg-info`。

## 归档而非直接删除

- `artifacts/runs/`、`artifacts/v2/runs/`；
- 旧验证集和评估结果；
- 旧模型 checkpoint；
- 旧版谱系和审核页面；
- 旧端口日志。

删除前必须确认没有审核页面、报告或训练脚本引用，并至少保留一份压缩归档。

