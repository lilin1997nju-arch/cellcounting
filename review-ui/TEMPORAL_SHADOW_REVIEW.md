# Temporal Shadow 独立审核界面

这是一个独立于现有审核页面的本地审核界面，面向 `artifacts/v3_shadow_tests/<run_id>/` 下的 Shadow 输出。

启动：

```powershell
cd E:\CM\cell-vision
.\.venv\Scripts\python.exe scripts\serve_v3_shadow_review.py --root artifacts\v3_shadow_tests --port 8788
```

打开 <http://127.0.0.1:8788/>。

功能：

- 按 Shadow run、plate、行为类型、Legacy 是否变化和审核状态筛选轨迹；
- 展示同一轨迹的 T0/T1/T2 图像、Legacy/V3 标签、概率和行为证据；
- 支持逐帧人工标签：单细胞、接触双细胞、≥3 细胞簇、杂质、无效、不确定；
- 对 `cell_to_debris` 轨迹使用统一轨迹标签，默认是“死细胞”；T0/T1/T2 只作为形态证据，不保存成混合的细胞 / 不确定 / 杂质序列；
- 支持接受 V3、保留 Legacy、保存人工标签、需要更多证据和跳过；
- 审核结果写入 `temporal_shadow_reviews.json`，不会修改原始预测 CSV 或现有 `annotations.db`。

快捷键：`← / →` 切换轨迹，`1` 接受 V3，`2` 保留 Legacy，`3` 保存人工逐帧标签。
