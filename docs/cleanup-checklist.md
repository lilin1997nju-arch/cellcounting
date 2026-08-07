# 上服务器前清理清单

这是“先归档、后删除”的清单。执行前应停止批处理和审核服务，并把归档目录复制到独立磁盘。

## 可以直接从部署包排除

- `.venv/`
- `__pycache__/`
- `.pytest_cache/`
- `.pytest-tmp/`
- `src/cellvision_local.egg-info/`
- `artifacts/cache/`
- `artifacts/*.log`

## 需要确认引用后归档

- `artifacts/runs/`
- `artifacts/v2/runs/`
- `artifacts/evaluation/` 中旧评估轮次
- `artifacts/validation/` 中已经冻结的旧验证结果
- 旧版 `review-ui/screening.*`
- `baseline.py`、`infer.py`、`tracking.py`、`lineage_engine.py`

## 绝不能直接删除

- 原始 TIFF 和 `sessions.idx`
- 人工标注数据库
- 当前训练集/验证集标注快照
- 当前生产模型
- 最终孔级报告
- 对应的运行配置和 `run_metadata.json`

