# QL2603 T1-1/T1-2 T0–T2 验证

两块板使用独立 artifact root，配置中的 `experiment.timepoint_directories` 直接指向
`sessions.idx` 解析出的三个 timestamp session，因此后续 Day7/Day14（T3/T4）不会被
读取。运行命令：

```powershell
.venv\Scripts\python.exe scripts/run_timed_validation.py `
  --config configs/ql2603_t1_1_validation.yaml --source-artifacts artifacts
.venv\Scripts\python.exe scripts/run_timed_validation.py `
  --config configs/ql2603_t1_2_validation.yaml --source-artifacts artifacts
```

每块板的 `artifacts/validation/ql2603_t1_*/validation_timing.json` 保存逐阶段耗时、
CUDA 设备、候选数和最终标签统计；`latest_v2_predictions.csv` 与
`latest_well_screening.csv` 是该板的验证结果。合并分析文件位于
`artifacts/validation/ql2603_t1_1_t1_2/`，但后续纳入训练时应继续以两个独立板级目录
注册，避免同名孔发生数据泄漏。
