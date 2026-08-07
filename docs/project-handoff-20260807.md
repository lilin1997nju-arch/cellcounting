# Cell Vision 项目交接文档

更新时间：2026-08-07  
工作目录：E:\CM\cell-vision  
Git状态：当前目录还没有初始化 .git，后续由项目负责人建立仓库。

这份文档用于下一次对话继续推进项目上线，不需要重新回顾全部历史聊天。

---

## 1. 当前运行状态

- Python虚拟环境：.venv
- 当前测试：121 passed
- 项目审核服务：http://127.0.0.1:8777/
- 当前服务项目：QL2603
- 当前QL2603项目板数：20块
- 当前服务只保留8777，不要再为每块板子开启新的端口。
- 健康检查：

~~~text
GET http://127.0.0.1:8777/api/health
GET http://127.0.0.1:8777/api/ready
~~~

- 当前服务重启脚本：

~~~powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\restart_project_server.ps1
~~~

如果PowerShell提示禁止运行脚本，原因是ExecutionPolicy，不是脚本语法错误。

---

## 2. 当前总体架构

~~~text
原始图片 + sessions.idx
        |
        v
session_index.py
        |
        v
时间点/板子/孔位 Manifest
        |
        +--> 末点生长快速筛选
        |       |
        |       +--> 无明显生长：直接生成孔级结论
        |       |
        |       +--> 有明显生长
        |
        v
T0/T1/T2候选生成
        |
        +--> CF/伪掩膜候选
        +--> dense raw候选补充
        +--> wall结构过滤与wall附近细胞救援
        |
        v
形态分类
        |
        v
实例分割与重复候选合并
        |
        v
单细胞/粘连2细胞/3+细胞团/杂质/待定
        |
        v
跨时间点时序证据
        |
        v
孔级结论和审核数据库
        |
        v
project_server -> review_server -> review-ui
~~~

主要代码目录：

~~~text
src/cellvision/
  config.py                  配置加载、路径和环境变量
  session_index.py           sessions.idx解析
  manifest.py                单板图片Manifest
  dense_candidates.py        高召回候选生成
  teaching.py                形态分类与自动标注
  multiplicity.py            单细胞/粘连/细胞团判定
  v2_instance_inference.py   V2实例分割
  v2_temporal_inference.py   V2时序证据
  temporal_appearance.py     跨帧物体外观比较
  well_screening.py          T0-T2孔级判定
  gated_screening.py         末点门控和最终报告
  late_growth_inference.py  后期时间点生长判断
  project_server.py          项目主页、板子挂载和任务接口
  review_server.py           单板审核服务，目前仍是大型单体模块
  task_queue.py              原子JSON任务状态存储
  worker.py                  通用worker生命周期基础结构
  provenance.py              运行元数据和输入/模型指纹

scripts/
  run_day14_gated_plate.py   单板末点门控完整流水线
  run_ql2603_project.py      QL2603多板顺序计算
  prepare_ql2603_project.py  生成项目manifest和板子配置
  restart_project_server.ps1 固定8777服务重启

review-ui/
  project-list.*             项目列表
  project-dashboard.*        项目/板子汇总
  auto-review.*              当前主审核页面
  teach.*                    形态教学
  doublet-teach.*            粘连细胞教学
  integrated-review.*        综合审核
~~~

---

## 3. 当前主计算流程

单板主流程位于：

scripts/run_day14_gated_plate.py

阶段顺序：

1. 读取末点（Day7、Day14或实际选择的后期时间点）；
2. 快速判断孔是否有明显生长；
3. 无生长孔跳过T0-T2深度计算；
4. 对阳性孔建立T0/T1/T2 Manifest；
5. 构建CF/伪掩膜候选；
6. dense raw候选增强；
7. 形态模型推理；
8. 自动审核轮次；
9. 多重性推理；
10. 综合训练/审核轮次；
11. V2实例分割；
12. V2时序证据；
13. 孔级早期结论；
14. Day7等后期代表性区域定位；
15. 输出最终96孔板报告。

QL2603项目批处理入口：

scripts/run_ql2603_project.py

目前采用顺序处理，20块板总耗时约2小时。主要瓶颈是候选生成、dense候选、实例分割和图像处理。

---

## 4. 已确定的业务判定规则

### 时间点

- Day0、Day1、Day2必须连续选择；
- 至少选择一个Day7或更晚的时间点作为末点；
- 默认使用已选择的最晚时间点作为末点；
- T3/T4是采集序号，不应再当作固定培养天数，报告优先显示Day标签；
- 部分旧代码仍硬编码T0-T4，需要继续泛化。

### 末点门控

- 末点无明显生长：孔标记为“无明显生长”，跳过T0-T2深度计算；
- 末点有明显生长：进入T0-T2识别和审核；
- 后期图片主要用于确认生长，不参与T0-T2候选的细粒度时序分类。

### 候选和实例

- 孔壁连续结构不应进入候选；
- 孔壁附近或与孔壁重合的真实细胞仍需保留；
- 一个完整细胞只能对应一个实例；
- 细胞团内部的局部峰不能再生成独立单细胞候选；
- 粘连2细胞和3+细胞团必须保留实际多重性。

### 时序

- 时序主要用于细胞/杂质不确定时的修正；
- 三帧物体内部高度稳定，应更强地偏向杂质；
- 细胞形态、面积、位移和增殖变化应支持细胞；
- 仅有背景相似不能触发时序；
- 某个时间点没有候选时，只比较近似位置的同一物体，不应把背景作为相似证据；
- 分裂证据只能基于时序修正前原本就是双细胞或细胞团的候选；
- 时序修正不能把粘连2细胞或细胞团统一改成单细胞；
- 疑似死细胞不应计入T0多细胞来源；
- T0只有一个细胞且被标记为疑似死细胞，但后续有明确分裂证据时，应移除死细胞标签。

### 孔级报告分类

目标分类包括：

- 无明显生长；
- 单细胞来源且有活性；
- 多细胞来源；
- 生长待确认；
- T0缺失但后期出现；
- T0-T2未观察到分裂；
- 证据不足/无法判断。

人工修改细胞或杂质、手动补充T0细胞、修改单细胞/粘连标签后，孔级结论必须同步重算。

---

## 5. 已完成的工程整理

### 配置和路径

- config.py已支持深度合并；
- 支持环境变量覆盖数据、artifact、模型、数据库和日志目录；
- 增加配置结构校验；
- shared_model_root已从硬编码路径改为可迁移路径；
- prepare_ql2603_project.py支持自定义数据根目录、artifact根目录和配置目录。

### 服务

- 项目服务固定默认使用8777；
- 项目板子改为懒加载，启动时不再一次性创建20个完整审核应用；
- 增加项目和单板的 /api/health、/api/ready；
- 增加稳定重启脚本；
- 远程绑定需要显式设置 --allow-remote 或 CELLVISION_ALLOW_REMOTE=1。

### 任务状态

- 增加TaskQueueStore；
- 支持queued、running、completed、error、cancelled；
- 增加任务详情和取消接口；
- 增加通用TaskWorker生命周期结构。

注意：目前worker还没有连接到真正的QL2603导入/推理执行器，当前任务队列主要完成了持久化和状态基础。

### 运行可追溯性

- Day14门控运行会写入 run_metadata.json；
- 记录配置摘要、输入文件指纹、模型文件指纹、阶段耗时和代码版本；
- 仍需要继续完善真正的模型registry和正式run目录结构。

### 已补充的部署文档

- docs/architecture.md
- docs/server-deployment.md
- docs/artifact-retention.md
- docs/cleanup-checklist.md
- docs/model-registry.md
- deploy/Dockerfile
- deploy/compose.yaml
- deploy/Caddyfile.example
- .env.example

---

## 6. 目前仍未解决的问题

### 高优先级

1. review_server.py仍超过10万字，是数据库、API、图像、训练和审核的单体模块；
2. 任务队列没有真正连接GPU worker；
3. 新任务解析后不能自动完成完整项目生成和推理；
4. 配置和项目manifest中仍有大量绝对Windows路径；
5. 没有用户认证、角色权限和审计账号体系；
6. SQLite没有正式迁移系统，暂不适合多用户并发审核；
7. 没有正式备份、恢复和失败重试机制；
8. 当前项目尚未建立Git和CI。

### 模型和业务逻辑

1. T0漏检仍是重点风险；
2. 孔壁附近细胞和孔壁残留结构仍需持续验证；
3. 细胞团内部重复候选虽然已有抑制逻辑，但需要固定指标验证；
4. 时序相似度必须继续从整块背景相似改为物体内部相似；
5. Day7/Day14成片细胞的代表性区域和阴影轮廓仍可能偏离实际；
6. 末点门控阈值需要用多板数据校准；
7. 需要固定报告目标级漏检率、重复候选率、孔壁误检率、T0/T1/T2召回率和孔级结论准确率。

### 前端

1. review_server.py仍负责太多计算，部分接口会同步阻塞；
2. 旧版 screening.* 和根目录旧页面暂未归档；
3. 放大图、图像缓存和多板切换仍可能有性能问题；
4. 手动修改后孔级结论同步重算需要继续逐例验证；
5. 当前UI还没有真正的任务进度、失败重试和取消展示。

---

## 7. 可归档但不要直接删除的内容

可排除出部署包：

- .venv/
- __pycache__/
- .pytest_cache/
- .pytest-tmp/
- src/cellvision_local.egg-info/
- artifacts/cache/
- 旧日志和预览图

需要先归档再确认引用：

- artifacts/runs/
- artifacts/v2/runs/
- 旧验证结果；
- 旧模型checkpoint；
- baseline.py；
- infer.py；
- tracking.py；
- lineage_engine.py；
- 旧版review-ui/screening.*。

绝对不能直接删除：

- 原始TIFF和sessions.idx；
- 人工审核数据库；
- 训练集和验证集人工标注；
- 当前生产模型；
- 最终报告；
- 与结果对应的配置和run_metadata.json。

---

## 8. 上线前推荐顺序

### 第一步：建立Git和基线

1. 初始化Git；
2. 提交源码、配置模板、部署文档和测试；
3. 不提交原始图像、缓存、大模型和真实审核数据库；
4. 建立稳定标签，例如 pre-server-refactor。

### 第二步：服务器目录和环境

建议分离：

~~~text
/srv/cellvision/app
/srv/cellvision/raw
/srv/cellvision/artifacts
/srv/cellvision/models
/srv/cellvision/db
/srv/cellvision/logs
~~~

Windows服务器也应保持同样的逻辑分层。

### 第三步：部署最小可运行服务

1. 设置.env；
2. 确认CUDA和Torch：

~~~powershell
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
~~~

3. 启动一个8777项目服务；
4. 检查 /api/health 和 /api/ready；
5. 使用反向代理和实验室VPN/内网访问。

### 第四步：补全任务执行器

推荐目标：

~~~text
项目UI -> API创建任务 -> 持久化队列 -> 单GPU worker
       -> 末点筛选 -> T0-T2推理 -> 报告 -> 审核
~~~

不要让训练或完整推理直接在HTTP请求中同步执行。

### 第五步：再做大规模模块拆分

优先拆分：

1. review_server.py的数据库层；
2. review_server.py的图像服务层；
3. review_server.py的训练/推理接口；
4. 项目级任务worker；
5. 旧谱系和V1代码迁移到legacy/。

---

## 9. 下一次对话建议直接使用的开场内容

~~~text
请先读取 E:\CM\cell-vision\docs\project-handoff-20260807.md。
当前目标是把项目部署到实验室服务器。先检查Git前的工作区、配置路径、
8777服务、CUDA环境和任务队列状态，然后按P0顺序推进，不要重新运行历史训练。
~~~

