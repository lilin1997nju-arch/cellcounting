# Cell Vision 检测、识别与分割融合模型分阶段开发文档

更新时间：2026-08-12  
适用仓库：`E:\CM\cell-vision`

## 1. 文档目的

本文档用于指导后续新对话在不影响当前生产流程的前提下，逐步训练和验证一个面向 96 孔板早期细胞图像的融合模型。

核心业务目标是：

1. 高召回地找到孔内真实细胞；
2. 排除外观接近细胞的杂质；
3. 排除连续孔壁、反光和孔壁纹理；
4. 保留贴在孔壁上或临近孔壁的真实细胞；
5. 对粘连双细胞和 3 个以上细胞团生成正确数量的实例；
6. 输出足够完整的实例 Mask，供形态和纹理特征提取；
7. 保留 V3 时序模型作为独立的最终证据层。

本项目不应一开始就重写成端到端实例分割。推荐从“整图候选检测器”开始，在固定验证集证明有效后，依次融合分类头和 Mask 头。

---

## 2. 当前生产基线

### 2.1 当前主流程

当前单板流程入口为 `scripts/run_day14_gated_plate.py`，主要顺序是：

```text
CF 连通域候选生成
→ 原图密集候选补充和孔壁 rescue
→ 形态分类：cell / debris / invalid
→ 多重性分类：single / touching_doublet / cluster_3plus
→ V2 候选点条件实例分割
→ V3 时序证据融合
→ 孔级结论
```

当前候选生成代码：

- `src/cellvision/pseudo_labels.py`
- `src/cellvision/dense_candidates.py`

当前分类与融合代码：

- `src/cellvision/teaching.py`
- `src/cellvision/multiplicity.py`

当前实例分割和时序代码：

- `src/cellvision/models/v2_instance_segmenter.py`
- `src/cellvision/v2_instance_inference.py`
- `src/cellvision/v2_temporal_inference.py`

### 2.2 当前模型资产

截至 2026-08-12，已知最新候选模型为：

- 形态分类头：`artifacts/projects/ql2603/plates/ql2603-t1-1/models/teaching_classifier.pt`
- 多重性分类头：`artifacts/projects/ql2603/plates/ql2603-t1-1/models/multiplicity_classifier.pt`
- 最新分割候选：`artifacts/v2/runs/v2-instance-20260812-120154/model.pt`
- 当前生产分割：`artifacts/v2/models/latest_instance_segmenter.pt`
- 当前时序模型：`artifacts/v2/models/latest_temporal_evidence.pt`

所有新模型在通过固定测试集之前只能写入新的 run 目录，不得覆盖以上生产模型。

### 2.3 当前审核数据快照

以下统计按每块板、每个 `candidate_id` 保留最后一次审核结论，统计日期为 2026-08-12：

| 数据 | 数量 |
|---|---:|
| 去重后的已审核候选 | 14,094 |
| 真实细胞 | 2,693 |
| └ single | 2,217 |
| └ touching_doublet | 387 |
| └ cluster_3plus | 89 |
| debris | 11,329 |
| 人工审核确认 invalid | 57 |
| uncertain | 15 |
| 人工补加的旧流程漏检细胞 | 29 |
| 有审核记录的板 | 12 |
| 临壁区域已审核候选 | 2,380 |
| 临壁区域已有候选、人工确认为细胞 | 1,109 |
| 临壁区域人工补加的漏检细胞 | 28 |
| 临壁区域人工确认细胞合计 | 1,137 |
| └ single | 906 |
| └ touching_doublet | 170 |
| └ cluster_3plus | 61 |
| 临壁区域 debris | 1,212 |
| 临壁区域人工审核确认 invalid | 53 |
| T1-1 中现有 Mask 审核记录 | 411 |

数据库中的主要监督来源为：

- `integrated_training_reviews`：已有候选的人工类别审核；
- `quick_missed_objects`：人工主动补加、旧候选流程漏掉的细胞；
- `annotations`：人工位置标注；
- `v2_mask_reviews`：接受、修改或拒绝的实例 Mask。

注意：数据库可能包含多个审核轮次。构建数据集时必须按 `updated_at` 和主键排序，然后按 `candidate_id` 保留最后一次有效决定，不能直接累计行数。

这里的 57 个 invalid 只代表人工审核后显式保存为 invalid 的候选，不代表可用的 invalid 训练样本总数。当前流程会让高置信度 invalid 不再进入审核队列，因此还存在大量自动 invalid。阶段 0 必须将它们按来源独立导出：

- `human_invalid`：人工确认，强监督；
- `deterministic_wall_invalid`：确定性孔壁规则产生，中强度监督；
- `classifier_high_confidence_invalid`：分类器伪标签，弱监督；
- `unreviewed_unknown`：没有可靠结论，不能当作 invalid。

自动 invalid 必须保留概率、候选来源、孔壁区域和 `label_origin`，并按板、孔、时间点、位置和孔壁角度去重/限额，避免大量相似孔壁纹理淹没真实细胞和杂质样本。

---

## 3. 数据监督的关键原则

### 3.1 按审核范围决定背景监督

用户采用的审核规则是：凡是已完成审核的孔/图像，如果孔壁上的细胞没有被旧流程检测到，会主动补加细胞标记。因此，能够确认已经完整审核的图像，其未标记孔壁区域可以作为背景/孔壁负监督，不必全部设为 ignore。

训练数据必须区分两种审核范围：

```text
完整审核图像/孔：
  已标细胞                      强正监督
  已审 debris / invalid         强负类别监督
  自动高置信度 invalid          按来源加权的负监督
  其余未标记孔壁和背景          背景/孔壁负监督

未完成完整审核的图像/孔：
  已审核位置附近                局部监督
  其他位置                      ignore
```

需要注意，旧审核主要围绕候选队列展开。如果数据系统无法可靠判断某个孔/图像是否已经完成整图审核，那么没有审核记录的位置仍可能是：

- 真正的背景；
- 未被旧候选发现的真实细胞；
- 没有进入审核队列的杂质或孔壁结构。

因此，阶段 0 应显式生成以下字段：

- `exhaustive_review`：该图像或孔是否完成穷尽审核；
- `missed_cell_marking_completed`：漏检细胞补标是否完成；
- `review_scope`：`full_image`、`full_well` 或 `candidate_only`；
- `supervision_region`：允许计算背景损失的区域。

优先从审核轮次状态和审核完成记录自动推导；如果现有数据库不能无歧义地推导，则建立单独的审核范围 manifest，不能仅凭“数据库存在一些审核行”推定整图已完整审核。

最终监督规则为：

```text
已审核 cell 中心附近              正监督
已审核 debris / invalid 中心附近  负类别监督
人工补标漏检细胞附近              强正监督
精确人工 Mask 区域                像素级监督
完整审核范围内其余区域            背景/孔壁负监督
非完整审核范围内其他区域          ignore，不计算负样本损失
```

这使已审核板能够为整图检测提供真正的背景监督，同时保护尚未完整审核的数据不被错误标成负样本。

### 3.2 标注映射

统一训练标签建议如下：

| 审核标签 | 检测 objectness | 形态类别 | 多重性类别 | Mask 监督 |
|---|---:|---|---|---|
| single | 1 | cell | single | 有审核 Mask 时启用 |
| touching_doublet | 1 | cell | touching_doublet | 有审核 Mask 时启用 |
| cluster_3plus | 1 | cell | cluster_3plus | 有审核 Mask 时启用 |
| debris | 可作为非细胞对象 | debris | ignore | 空 Mask 或不训练 Mask |
| invalid | 可作为非细胞对象 | wall/invalid | ignore | 空 Mask 或孔壁监督 |
| deterministic_wall_invalid | 按权重作为非细胞对象 | wall/invalid | ignore | 可提供孔壁监督 |
| classifier_high_confidence_invalid | 弱负监督 | wall/invalid | ignore | 默认不提供精确 Mask |
| uncertain | ignore | ignore | ignore | ignore |
| quick_missed_objects 中的细胞 | 1 | cell | 对应审核类别 | 没有 Mask 时只做点监督 |
| v2_mask_reviews accepted/edited | 1 | cell | 继承最终审核类别 | 使用最终审核 Mask |
| v2_mask_reviews rejected | 0 或非细胞 | 继承可用类别 | ignore | 全零实例 Mask |

`debris` 和 `invalid` 不应简单合并：杂质可能位于孔内并且外形接近细胞，而 invalid 主要表示孔壁或不可用结构。模型可以共享骨干，但应保留不同分类目标或至少保留独立的孔壁概率头。

### 3.3 临壁定义

第一版沿用当前评估口径：

```text
radial_fraction >= 0.40
```

同时保留更细的 `candidate_zone`：

- `well_interior`
- `wall_cell_buffer`
- `wall_cell_rescue`
- `wall_residual`

训练和评估必须单独报告临壁数据，不允许只报告全体平均指标。当前临壁真实细胞基线应使用 1,137 个：1,109 个来自已有候选的人工确认，28 个来自人工主动补标。后者代表旧候选流程的真实漏检，应提高采样权重并单独报告恢复率。

### 3.4 数据划分

必须以板为最小划分单位，并保证同一孔、同一对象的 T0/T1/T2 不跨集合。

当前已经冻结的验证板必须继续保持隔离：

- `ql2603-t1-2`
- `ql2603-t4-2`

这两块板不得参与候选检测、分类、Mask 或阈值校准训练。后续可再冻结至少一块此前未参与开发的板作为最终盲测集。

---

## 4. 目标模型架构

### 4.1 推荐的长期结构

```text
输入图块
  ├─ 原始灰度图
  ├─ CF 图或 CF 概率图
  └─ 动态孔壁距离/先验图
          ↓
      共享特征骨干
          ├─ 中心/objectness 检测头
          ├─ cell/debris/wall 分类头
          ├─ single/doublet/cluster 多重性头
          ├─ 尺寸/直径回归头
          └─ 候选条件实例 Mask 头
                    ↓
              实例合并与去重
                    ↓
               V3 时序证据
```

Mask 头不应在第一阶段强行启用。现有类别和点标注远多于精确 Mask，先训练检测和分类可以更充分利用已有审核数据。

### 4.2 输入图块

建议从以下配置起步，并通过验证集调整：

- 图块大小：512×512 或 768×768；
- 图块重叠：64～128 像素；
- 保持原始分辨率，不直接把整孔缩小到细胞难以辨认的尺度；
- 输入通道：归一化原图、CF 图、动态孔壁先验图；
- 图像增强：亮度、对比度、轻微噪声、旋转、翻转；
- 禁止会破坏孔壁几何关系的强非刚性变换。

### 4.3 第一版检测输出

推荐使用 CenterNet 风格的中心热图，而不是只输出框：

- `cell_center_heatmap`
- `debris_center_heatmap`
- `wall_invalid_heatmap` 或独立 `wall_probability`
- `diameter/size`
- 可选 `single/doublet/cluster` logits

点检测更符合现有数据结构，也方便与当前候选点条件 V2 分割模型衔接。

### 4.4 孔内与临壁区域的模型设计

当前不开发两套完全独立的识别模型。推荐使用一套共享骨干和两个轻量区域专家头：

```text
原图 + CF 图 + 动态孔壁距离/方向先验
                  ↓
              共享骨干
          ┌───────┴───────┐
       孔内专家头       临壁专家头
  cell/debris/background  cell/debris/wall
          └───────┬───────┘
             距离加权融合
```

采用平滑过渡而不是以 `radial_fraction = 0.40` 硬切换：

- 孔内区域主要使用孔内头；
- 临壁过渡区融合两个头；
- 孔壁区域主要使用临壁头；
- 贴壁细胞、人工补标漏检细胞和分类冲突样本进入临壁困难样本池。

共享骨干可以使用全部 2,693 个已有候选人工细胞样本学习通用细胞特征，临壁头则重点使用 1,137 个临壁细胞、1,212 个临壁 debris、人工确认 invalid 以及分层抽样的自动 invalid。只有当固定测试表明共享结构存在稳定的负迁移，才考虑拆成两套完全独立模型。

---

## 5. 分阶段开发计划

## 阶段 0：数据集、审计和固定评估

### 目标

建立可重复的数据导出器、按板划分的数据集，以及独立于生产结果的固定评估工具。此阶段不训练新模型。

### 必须实现

1. 新增统一审核数据导出脚本，读取所有指定板的 SQLite 数据库；
2. 按 `candidate_id` 保留最终审核结果；
3. 合并候选 CSV 中的位置、来源、`radial_fraction`、`candidate_zone`；
4. 合并 `quick_missed_objects`；
5. 合并最终人工 Mask，并区分 accepted、edited、rejected；
6. 导出所有自动 invalid，并区分确定性规则与分类器伪标签；
7. 建立图像/孔级审核范围 manifest，记录 `exhaustive_review` 和 `review_scope`；
8. 输出训练、验证、测试 manifest；
9. 为每个图块输出监督有效区 `supervision_mask` 或等价 ignore 标志；完整审核范围内允许背景监督，其他区域保持 ignore；
10. 生成数据审计报告，检查图像缺失、重复样本、坐标越界、标签冲突和跨集合泄漏；
11. 冻结基线候选结果和基线指标。

### 建议新增文件

```text
configs/fusion_detector_v1.yaml
scripts/export_fusion_training_dataset.py
scripts/audit_fusion_training_dataset.py
src/cellvision/fusion_dataset.py
tests/test_fusion_dataset.py
```

### 建议输出目录

```text
artifacts/v2/datasets/fusion-detector-v1/
  dataset_manifest.csv
  train.csv
  validation.csv
  test.csv
  mask_index.csv
  review_scope.csv
  automatic_invalid.csv
  audit.json
  split_manifest.json
```

### 阶段完成条件

- 每个监督样本可以回溯到板、孔、时间点、候选 ID、数据库记录和原图；
- T1-2、T4-2 没有出现在训练集；
- `uncertain` 和非完整审核范围的未知区域不会产生负监督；
- 完整审核范围内的未标记区域能够进入背景/孔壁监督；
- 1,137 个临壁人工确认细胞可追溯，其中28个标记为旧流程漏检；
- 自动 invalid 均保留来源和训练权重，没有冒充人工真值；
- 数据审计报告中的跨集合泄漏为 0；
- 统计数量能解释与本文件快照的差异，例如新增审核或标签更新。

## 阶段 1：整图候选检测器，影子模式运行

### 目标

用训练模型补充或替代部分 CF/密集规则候选，但暂不改变生产输出。模型只负责提出候选位置和分数，后续仍使用现有分类器、V2 Mask 和 V3。

### 推荐实现

1. 训练中心热图检测器；
2. 对完整审核范围使用完整检测损失，对其他未知区域使用 ignore loss；
3. 对 `quick_missed_objects` 提高采样权重；
4. 对临壁 cell、临壁 debris、人工 invalid 和自动 invalid 分层采样；
5. 使用 focal loss 或等价方式处理正负不平衡；
6. 使用多尺度图块推理和跨图块半径去重；
7. 输出独立候选来源 `learned_full_image_detector`；
8. 与旧候选取并集后送入当前分类和分割流程；
9. 不允许模型输出直接覆盖 `morphology_candidates.csv`，必须写入独立 shadow run。
10. 第一版先使用共享骨干和孔壁位置输入；分类融合阶段再启用孔内/临壁专家头，不维护两套完全独立模型。

### 输出

```text
artifacts/v2/runs/fusion-detector-<timestamp>/
  model.pt
  config.yaml
  run_metadata.json
  metrics.json
  predictions/<plate>/detector_candidates.csv
  comparisons/<plate>/candidate_comparison.csv
```

### 核心指标

- 人工真细胞候选召回率；
- 临壁真细胞候选召回率；
- single、doublet、cluster 分层召回率；
- 人工补标漏检细胞恢复率；
- 每孔候选数量；
- 每孔新增假阳性数量；
- 推理耗时；
- 新模型相对旧候选的新增真阳性和新增假阳性。

### 晋级门槛

至少满足：

1. 固定测试板总体细胞候选召回不低于旧 CF+密集候选并集；
2. 临壁细胞召回不低于旧流程，并优先追求提升；
3. 人工补标漏检细胞恢复率明显高于旧流程；
4. 新增候选数量没有使后续 V2 推理时间不可接受；
5. 所有指标按板报告，并提供 bootstrap 置信区间或逐板差值，不能只报告汇总均值。

阶段 1 通过后，仍先保留旧候选与新候选并行一段时间。

## 阶段 2：融合候选检测和类别识别

### 目标

在共享骨干上增加形态和多重性分类头，使一次图块推理同时输出位置和类别。当前两个分类器继续作为对照，不立即删除。

### 训练任务

- 检测：是否存在可审核对象及其中心；
- 形态：cell / debris / wall-invalid；
- 多重性：single / touching_doublet / cluster_3plus；
- 尺寸：直径或局部尺度；
- 可选质量头：是否需要 V2 Mask 精细推理或人工审核。

### 损失设计

总损失建议为：

```text
L = λdet Lcenter
  + λcls Lmorphology
  + λmulti Lmultiplicity
  + λsize Lsize
```

每个头只在拥有对应人工标签的位置计算损失。不能用模型自身的历史预测作为高权重真值。

### 分类安全规则

- 高置信度 cell：进入 V2 Mask；
- cell/debris 冲突或低置信度：进入 V2 Mask或审核；
- 高置信度 debris：可跳过 Mask，但保留抽样审计；
- 高置信度 wall-invalid：可跳过 Mask；
- 临壁候选只有在 wall 证据强、cell 证据弱时才允许硬跳过；
- doublet/cluster：必须进入多种子或实例拆分分支。

### 晋级门槛

- 真细胞被判成 invalid 的比例不高于当前分类器；
- cell/debris 宏平均 F1 不下降；
- 临壁 cell 的召回率不下降；
- 每孔进入 V2 Mask 的候选数量下降或保持可控；
- 在冻结测试板上，新融合分类头相对两个当前分类头具有明确收益，或在同等准确率下降低总耗时。

只有满足以上条件后，才允许逐步关闭当前独立分类头。

## 阶段 3：接入共享 Mask 头

### 目标

将检测、分类和实例 Mask 共享特征骨干，但继续保留当前点条件 V2 分割作为精修器和回退路径。

### 数据使用

- accepted Mask：精确正监督；
- edited Mask：最高价值的边界纠错监督，可提高采样权重；
- rejected Mask：非细胞或无效实例的全零监督；
- 没有人工 Mask 的 cell：只用于检测和分类，不强制生成像素级真值；
- 没有人工 Mask 的区域：Mask loss 默认 ignore；完整审核范围只提供检测背景监督，不能据此虚构像素级实例边界。

### 推荐结构

第一版不要直接做全图语义 Mask。建议使用：

```text
共享图块骨干
→ 中心检测
→ 针对每个中心提取 ROI 特征
→ 候选条件实例 Mask 头
```

对于 doublet/cluster，使用多个中心种子或中心偏移头，避免一个粘连区域只输出一个整体轮廓。

### Mask 指标

- IoU 和 Dice；
- IoU 的中位数与第 10 百分位数；
- 轮廓完整率；
- 中心是否位于最终 Mask 内；
- 面积异常缩小率；
- 粘连实例拆分召回率；
- 相邻细胞错误合并率；
- 孔壁误覆盖率；
- 人工编辑率和拒绝率。

不能只比较平均 IoU。当前问题集中在少量严重失败样本，第 10 百分位数、失败率和人工回退率更重要。

### 晋级门槛

- 固定 Mask 测试集的平均和低分位 IoU 不低于当前最新 V2；
- 粘连样本的实例数正确率提高；
- 临壁细胞的完整轮廓率不下降；
- 杂质和孔壁产生有效 Mask 的比例不增加；
- 严重失败样本能够自动回退到当前 V2 精修器。

## 阶段 4：选择性精修和生产影子部署

### 目标

形成计算量可控的融合推理流程：大多数简单样本一次完成，困难样本使用当前高精度 V2 精修。

### 建议流程

```text
融合模型整图/图块推理
→ 高置信度 debris/wall：排除
→ 高置信度单细胞且 Mask 质量高：直接采用
→ 临壁、粘连、边界冲突、低置信度：调用 V2 精修
→ 实例去重和层级合并
→ V3 时序验证
```

### 自动精修触发条件

- 临壁或与孔壁高度重叠；
- doublet/cluster 概率高；
- Mask 触碰 ROI 边缘；
- 中心不在 Mask 内；
- Mask 多连通域；
- 面积、圆度或直径异常；
- 分类头与 Mask presence 冲突；
- T0/T1/T2 对象匹配冲突；
- 模型置信度落在预设灰区。

### 生产前要求

- 至少完成六板 shadow 对比；
- 结果在现有 8777 审核界面可逐对象查看；
- 审核界面必须显示旧候选、新候选、旧分类、新分类、旧 Mask、新 Mask及最终采用路径；
- 生产模型替换必须有备份、模型指纹、配置指纹和一键回退；
- 不允许直接覆盖原始审核数据库或原始生产结果。

## 阶段 5：主动学习闭环

### 目标

使用新模型与旧流程的差异持续扩充训练队列，重点获得旧流程完全漏掉的细胞和困难孔壁样本。

### 审核队列优先级

1. 新模型发现、旧候选没有发现的位置；
2. 旧候选发现、新模型拒绝的位置；
3. 新旧分类冲突；
4. 临壁和贴壁候选；
5. doublet/cluster 或多中心冲突；
6. 新旧 Mask IoU 低或实例数不同；
7. 模型置信度接近决策阈值；
8. 每轮随机抽取部分高置信度排除样本，用于监控系统性漏检。

每轮审核后：

```text
冻结旧测试集
→ 追加新训练数据
→ 重新训练 challenger
→ 在相同测试集比较
→ 通过门槛后进入 shadow
→ 不通过则保留当前生产模型
```

---

## 6. 统一评估协议

## 6.1 候选层

候选点与人工真值中心在规定半径内匹配，并采用一对一匹配。至少报告：

- overall cell recall；
- near-wall cell recall；
- single/doublet/cluster recall；
- manual-missed recovery；
- proposals per well；
- duplicate proposal rate；
- wall false proposals per well；
- debris false proposals per well。

## 6.2 分类层

- cell recall；
- cell precision；
- debris recall/precision；
- invalid recall/precision；
- cell → invalid 错误率；
- near-wall cell → invalid/debris 错误率；
- macro F1；
- 校准误差和可靠性曲线。

## 6.3 实例层

- IoU、Dice；
- 第 10/25/50 百分位 IoU；
- complete-contour pass rate；
- touching split recall；
- merge error、over-split error；
- near-wall contour pass rate；
- mask rejection/edit rate。

## 6.4 孔级最终结果

- 单细胞来源准确率；
- 多细胞来源准确率；
- 无明显生长准确率；
- undetermined 比例；
- 每孔假阳性数量；
- 因漏检导致的孔级结论错误数。

所有指标必须同时给出逐板结果和总体结果。训练集指标不能作为模型晋级依据。

---

## 7. 实验与模型资产规范

每次训练必须生成独立目录：

```text
artifacts/v2/runs/<stage>-<timestamp>/
  model.pt
  config.yaml
  run_metadata.json
  dataset_fingerprint.json
  split_manifest.json
  metrics.json
  per_plate_metrics.csv
  failures.csv
  predictions/
```

`run_metadata.json` 至少记录：

- Git commit；
- 训练配置；
- 训练、验证、测试板列表；
- 数据集指纹；
- 输入图像指纹；
- 初始 checkpoint；
- 随机种子；
- CUDA、PyTorch 和设备信息；
- 训练开始和结束时间；
- 最佳 epoch 和选择指标。

模型晋级遵循 challenger/champion 机制：

- challenger 只存在于 run 目录；
- shadow 验证通过后才允许复制到模型 registry；
- 替换生产模型前备份当前 checkpoint 和配置；
- 回退不得依赖重新训练。

---

## 8. 工程约束和测试

1. 不修改原始 TIFF、sessions.idx 和人工审核数据库；
2. 数据导出和训练过程只读源数据库；
3. 不覆盖现有 `morphology_candidates.csv`；
4. 所有模型推理输出写到新 run 目录；
5. 保留旧候选、旧分类、旧 Mask 作为逐对象对照；
6. 新增代码需要单元测试；
7. 至少测试以下边界情况：
   - 完整审核范围内未标记区域可作为检测背景；
   - 非完整审核范围的未知区域被正确 ignore；
   - 审核范围无法确定时默认采用保守的 `candidate_only`；
   - 同一候选多个审核轮次只取最终结论；
   - 同一孔的时间点不跨数据集；
   - 临壁坐标和孔壁先验对齐；
   - 图块边界候选不重复；
   - 人工补标漏检细胞进入正样本；
   - 28 个临壁漏检补标被识别为高价值正样本；
   - 自动 invalid 保留来源、置信度和弱监督权重；
   - rejected Mask 不作为正实例；
   - holdout 板不会进入训练 cache。

当前工作区可能存在用户未提交修改。新对话开始工作时必须先运行 `git status --short`，只修改本阶段明确涉及的文件，不得覆盖或提交无关改动。

---

## 9. 推荐开发顺序和停止条件

严格按以下顺序执行：

1. 阶段 0：数据导出、ignore 监督、固定评估；
2. 阶段 1：检测器 shadow；
3. 阶段 2：融合分类头；
4. 阶段 3：融合 Mask 头；
5. 阶段 4：选择性精修与 shadow 部署；
6. 阶段 5：主动学习循环。

以下情况必须停止晋级并分析失败样本：

- 总体或临壁细胞召回下降；
- 新模型只在训练板提升、冻结板不提升；
- 每孔假阳性数量显著增加；
- 人工补标漏检细胞恢复率没有提高；
- Mask 平均 IoU 提高但低分位严重失败增加；
- 粘连实例数正确率下降；
- 运行时间或显存超出可接受范围；
- 评估集污染或数据指纹无法复现。

---

## 10. 新对话启动提示词

建议新开对话后先只执行阶段 0。可直接复制以下内容：

```text
请在 E:\CM\cell-vision 仓库中继续融合模型开发。

首先完整阅读：
E:\CM\cell-vision\docs\fusion_model_staged_development.md

本次只执行文档中的“阶段 0：数据集、审计和固定评估”，不要开始训练，不要替换或覆盖任何生产模型，也不要修改原始审核数据库和 morphology_candidates.csv。

开始前先检查 git status，保留所有已有且与本任务无关的用户修改。请实现：
1. 统一导出 integrated_training_reviews、quick_missed_objects、annotations 和 v2_mask_reviews；
2. 每个 candidate_id 只保留最终有效审核；
3. 合并候选位置、radial_fraction、candidate_zone 和 candidate_source；
4. 建立图像/孔级审核范围：用户在已完整审核图像中会主动补标所有漏检的孔壁细胞；完整审核范围内未标记区域可作为背景，其他未知区域必须 ignore；
5. 导出人工 invalid、确定性孔壁 invalid 和分类器高置信度 invalid，保留 label_origin、概率和训练权重，不能把自动标签冒充人工真值；
6. 按板划分训练/验证/测试，ql2603-t1-2 和 ql2603-t4-2 必须保持冻结；
7. 生成可复现的数据 manifest、审核范围 manifest、数据指纹和审计报告；
8. 增加测试，验证无跨板泄漏、最终审核去重、完整/非完整审核监督规则、人工漏检正样本和自动 invalid 来源；
9. 核对临壁人工确认细胞基线：合计1,137个，其中1,109个是已有候选经人工确认，28个是人工补标漏检；
10. 运行相关测试并汇报实际样本数量、标签分布、临壁分布和所有发现的数据问题。

完成阶段 0 后停止，先向我汇报结果和阶段 1 的训练建议，等待我确认后再训练。
```

阶段 0 完成并确认后，阶段 1 的新对话提示词：

```text
请继续 E:\CM\cell-vision 的融合模型开发。完整阅读 docs/fusion_model_staged_development.md，并检查阶段 0 已生成的数据审计、split manifest 和测试结果。

本次只执行“阶段 1：整图候选检测器，影子模式运行”。完整审核范围使用完整检测监督，其他未知区域必须 ignore；自动 invalid 按来源和置信度加权。使用共享骨干和孔壁距离/方向输入，不开发两套完全独立模型。ql2603-t1-2 和 ql2603-t4-2 只能评估，不能训练或调参。训练输出必须写入新的 artifacts/v2/runs/fusion-detector-<timestamp> 目录，不能覆盖生产候选、分类、分割或时序模型。

实现训练、图块推理、跨图块去重、旧/新候选逐点比较和逐板评估。重点报告总体细胞召回、临壁细胞召回、single/doublet/cluster 召回、人工漏检恢复率、每孔新增假阳性、候选数和耗时。完成 shadow 对比后停止，不要自动部署。
```

---

## 11. 本路线的最终判定

本路线不是立即把所有模型合成一个不可拆分的网络，而是逐步融合：

```text
近期：训练型候选检测器 + 当前分类器 + 当前 V2 Mask + V3
中期：共享检测/分类骨干 + 当前 V2 Mask + V3
长期：共享检测/分类/Mask 模型 + 困难样本 V2 精修 + V3
```

这种顺序能够最大程度使用现有 14,094 条类别审核数据，同时避免因精确 Mask 数量较少和整图标注不完整而过早训练出有系统性漏检的端到端模型。
