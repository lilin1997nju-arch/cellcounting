# 模型登记规范

部署到服务器时，模型文件不能只依赖 `latest_*.pt` 这种可变名称。每个生产模型至少登记：

```json
{
  "model_id": "v2-instance-20260806-01",
  "task": "instance_segmentation",
  "path": "models/v2-instance-20260806-01.pt",
  "sha256": "...",
  "training_data": ["QL11111", "QL2202", "T1-1"],
  "validation_summary": "evaluation/v2_metrics.json",
  "config_digest": "...",
  "created_at": "2026-08-06T00:00:00Z",
  "status": "production"
}
```

推荐目录：

```text
models/
  registry.json
  v2-instance-20260806-01.pt
  v2-temporal-20260806-01.pt
  morphology-20260806-01.pt
```

`latest_*`可以作为兼容软链接或指针，但报告必须记录真正的模型ID和SHA256。

