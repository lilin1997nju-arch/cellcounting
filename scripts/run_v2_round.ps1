param(
    [string]$Python = ".\.venv\Scripts\python.exe"
)

$instanceModel = "artifacts/v2/models/latest_instance_segmenter.pt"
$temporalModel = "artifacts/v2/models/latest_temporal_evidence.pt"
$configs = @(
    "configs/default.yaml",
    "configs/ql2202_validation.yaml",
    "configs/a12_22_training.yaml"
)

& $Python -m cellvision train v2-instance-segmenter --config configs/v2_training.yaml
& $Python -m cellvision train v2-temporal-model --config configs/v2_training.yaml
foreach ($config in $configs) {
    & $Python -m cellvision infer-v2 --config $config --checkpoint $instanceModel
    & $Python -m cellvision infer-v2-temporal --config $config --checkpoint $temporalModel
    & $Python -m cellvision evaluate-v2 --config $config
}
& $Python scripts/aggregate_v2_metrics.py
