# Cell Vision 当前架构与整理约定

## 目标

Cell Vision 是一个面向 96 孔板多时间点图像的本地视觉分析系统。推荐的生产链路是：

```text
原始数据/sessions.idx
        ↓
导入、校验、可迁移 Manifest
        ↓
末点生长快速筛选
        ├─ 无明显生长：直接写入孔级报告
        └─ 有明显生长：执行 T0/T1/T2 候选、实例、形态、时序和孔级判定
        ↓
版本化结果与审核数据库
        ↓
项目级审核服务
```

## 代码分层目标

当前代码仍然以研究迭代为主，下一步按下面的边界逐步整理，不要求一次性重写：

| 层 | 职责 | 当前来源 |
|---|---|---|
| domain | Well、Timepoint、Candidate、Review、WellConclusion、Job 数据结构 | 目前散落在多个模块和 DataFrame 中 |
| ingest | `sessions.idx`、文件夹、TIFF 和时间点校验 | `session_index.py`、`manifest.py` |
| pipeline | 可缓存的阶段函数和耗时记录 | `scripts/run_day14_gated_plate.py`、各推理模块 |
| models | 模型训练、checkpoint、推理 | `train*.py`、`v2_*`、`model_inference.py` |
| decision | 时序证据、孔级结论、末点门控 | `v2_temporal_inference.py`、`well_screening.py`、`gated_screening.py` |
| storage | 数据库、artifact、版本和迁移 | `review_server.py` 中仍有大量实现 |
| api | 轻量HTTP接口 | `review_server.py`、`project_server.py` |
| workers | 长时间训练和批处理 | 目前缺少正式执行器 |
| legacy | V1、旧谱系和旧审核页面 | `baseline.py`、`infer.py`、`lineage_engine.py` 等 |

## 当前运行原则

1. 项目服务使用一个稳定端口，默认 `8777`。
2. 项目服务懒加载板子审核应用，不在启动时一次性加载所有板子。
3. 原始数据只读；中间结果、模型和审核数据写入 artifact/数据库目录。
4. 末点门控优先于 T0-T2 深度计算。
5. 每次正式计算都应保存输入指纹、配置、模型版本、代码版本和阶段耗时。

## 暂不删除的兼容模块

下列模块已经不是主流程的核心，但仍可能被历史脚本或旧审核结果读取：

- `lineage_engine.py`、`tracking.py`
- `baseline.py`、`infer.py`
- `segmentation.py`、`candidates.py`
- `review-ui/screening.*`、旧版 `index.html/app.js/styles.css`

先迁移到 `legacy/` 或建立引用清单，再删除。不能直接删除历史审核数据库和人工标注。

