# Day14 门控的96孔筛选流程

## 目标

先用计算量较低的 Day14 成片生长检测排除没有生长起来的孔，只对阳性孔运行 T0～T2 的实例识别、分类与时序校正。Day7 只用于快速寻找代表性高密区域，不参与来源结论，也不绘制单细胞轮廓。

## 执行顺序

1. **Day14 生长门控**
   - 高反差 CF 图用于面积、连通区域和内部厚度计算。
   - 原始灰度图保留为人工核查证据。
   - 未达到“明显成片生长”阈值：直接输出“无明显生长”，停止该孔后续计算。
   - Day14 缺失或不可读：不得当作阴性，进入“待确定”。
2. **T0～T2 深度计算（仅 Day14 阳性孔）**
   - 沿用 V2 完整实例分割、孔壁过滤、实例去重、单帧分类和三帧时序校正。
   - 来源只由 T0 的最终完整实例决定。
   - `single=1`、`touching_doublet=2`、`cluster_3plus>=3` 个细胞单位。
3. **Day7 代表区域定位（仅 Day14 阳性孔）**
   - CF 图降采样到最长边不超过 1024 px。
   - 用约 1000 px 的滑动窗口计算总前景密度和厚前景密度。
   - 输出最多3个非重叠候选框，排名第1的框作为默认代表区域。
   - 不运行实例标注、不运行时序模型，不改变孔级结论。
4. **96孔总览**
   - 每孔仅有一个互斥主类别。
   - 待确定孔同时显示细分原因。
   - A1作为阳性对照单独显示，不纳入样本来源统计。

## 孔级判定优先级

1. A1或明确指定的对照：`positive_control`。
2. Day14 缺失/不可读：`undetermined / day14_missing_or_unreadable`。
3. Day14 无明显成片生长：`no_obvious_growth`，跳过T0～T2和Day7。
4. Day14 阳性但T0～T2尚未计算：`undetermined / early_results_not_computed`，并进入深度计算队列。
5. T0～T2缺图、图像质量失败、T0有细胞/杂质冲突：分别进入对应待确定子类。
6. T0细胞单位为0：
   - T1/T2出现细胞：`undetermined / t0_missing_later_detected`。
   - T1/T2也没有细胞：`undetermined / day14_growth_but_no_early_cell`。
7. T0细胞单位大于1，或有多个独立细胞实例：`multi_cell_origin`。
8. T0恰好一个细胞：
   - 有可信早期分裂证据：`single_cell_origin`。
   - 时序关联冲突：`undetermined / temporal_link_conflict`。
   - T0～T2未分裂：`undetermined / t0_t2_no_division`。

## 不改变主类别的附加证据

- 存在杂质。
- 存在疑似死细胞。
- Day7 未找到高密代表区域。
- 人工复核比例和单帧/时序置信度。

这些内容只作为说明和审核优先级，不应把单细胞来源强行改成多细胞来源，也不能用 Day7 阴性推翻 Day14 阳性。

## 输出

- `day14_positive_early_compute_queue.csv`：仅含需要运行T0～T2的Day14阳性样本孔。
- `plate_overview.csv`：96孔扁平总览。
- `plate_overview.json`：包含汇总计数、待确定原因计数和每孔证据，供审核页面96孔对话框读取。
- `day7_regions_json`：Day7代表区域矩形坐标，不包含细胞标注。

## 命令示例

```powershell
.\.venv\Scripts\cellvision.exe build-gated-screening `
  --day14-csv artifacts\day14_screening\ql2603\ql2603_day14_well_screening.csv `
  --group-id "QL2603 T1-2" `
  --early-screening-csv artifacts\validation\ql2603_t1_2\predictions\latest_well_screening.csv `
  --sessions-csv artifacts\ingest\ql2603_sessions.csv `
  --output-dir artifacts\gated_screening\ql2603_t1_2
```

第一次运行时可以不提供 `--early-screening-csv`，先生成 Day14 阳性深度计算队列；T0～T2推理完成后再提供早期结果重新运行，即得到最终孔级结论。
