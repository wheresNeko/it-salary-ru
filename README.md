# 俄罗斯 IT 岗位薪资预测

[![tests](https://github.com/wheresNeko/it-salary-ru/actions/workflows/tests.yml/badge.svg)](https://github.com/wheresNeko/it-salary-ru/actions/workflows/tests.yml)

基于俄罗斯国家就业门户开放数据的机器学习项目。目标是从招聘信息中预测 IT 岗位的薪资水平，并量化「哪些因素真正决定薪资」。

> **English version: [README.en.md](README.en.md)**
>
> **📄 最终报告（中英双语）：[`docs/final_report.zh.md`](docs/final_report.zh.md) · [`docs/final_report.md`](docs/final_report.md)** —— 把四段流程的发现汇总成一份可提交的文档。下面的 README 是项目说明与操作手册。

---

## 研究问题

> 俄罗斯 IT 就业市场中，哪些因素决定薪资水平？能否仅从岗位文本和结构化属性中准确预测薪资？

子问题：

1. 薪资信号有多少来自结构化属性（地区、学历、经验、企业、工作制），多少来自自由文本需求？门户不提供结构化技能字段，需要从文本中挖掘多少信息？
2. 在控制地区和学历后，IT 岗位相对其他行业的薪资溢价有多大？
3. 预测精度是否值得使用神经网络，还是更简单的模型表现相同？

子问题 3 是刻意的：验证阶段要给出可辩护的结论，而不只是一串测试名称。

---

## 为什么是这个数据源

**主数据源**：[Трудвсем / «Работа России»](https://trudvsem.ru/opendata/api)（俄罗斯国家就业门户）官方开放数据 API

| 属性 | 值 |
|---|---|
| 端点 | `https://opendata.trudvsem.ru/api/v1/vacancies` |
| 运营方 | 俄罗斯联邦劳动就业局（Роструд） |
| 认证 | **不需要** |
| 格式 | JSON |
| 全国岗位总量 | **522,303** |
| 费用 | 免费 |
| 法律地位 | 作为官方开放数据发布，明确授权复用 |

最初考虑的是 hh.ru（俄罗斯最大招聘网站）。**它在 2026 年 4 月关闭了公开的岗位搜索 API**：`/vacancies` 对未授权请求返回 `403`，密钥只发给通过审核的认证雇主和招聘服务商。实测确认：hh.ru 的 `/areas`、`/professional_roles` 返回 200，但 `/vacancies`、`/vacancies/{id}`、`/employers` 全部 403。

这不只是「换个源」的问题 —— 依赖爬取大型商业招聘网站的方案在法律和技术上都站不稳，而开放数据的授权是明确的。

---

## 数据长什么样

每条岗位记录包含 **31 个字段**：

| 分组 | 字段 |
|---|---|
| 薪资 | `salary`（自由文本）、`salary_min`、`salary_max`、`currency` |
| 岗位 | `job-name`、`qualification`、`schedule`、`employment`、`code_profession`、`typicalPosition`、`work_places` |
| 文本 | `requirements`、`duty`（俄语自由文本）、`skills`（**81% 为空**） |
| 要求 | `requirement.education`、`requirement.experience` |
| 企业 | `company.name`、`inn`、`ogrn`、`kpp`、`site`、行业 |
| 地区 | `region.name`、`region.region_code`、`addresses`（含经纬度） |
| 时间 | `creation-date`、`date_modify` |

### 规模（实测）

IT 子集通过 `text=` 关键词 × `region_code` 切片获得：

| 关键词 | 全国 | 彼尔姆 (59) | 莫斯科 (77) | 斯维尔德洛夫斯克 (66) |
|---|---:|---:|---:|---:|
| 1С | 9,042 | 133 | 592 | 327 |
| программист | 2,089 | 23 | 220 | 97 |
| инженер-программист | 996 | 15 | 83 | 46 |
| системный администратор | 894 | 9 | 44 | 16 |
| разработчик | 742 | 11 | 167 | 37 |
| аналитик данных | 511 | 19 | 118 | 11 |
| python | 431 | 5 | 144 | 21 |
| тестировщик | 79 | 1 | 19 | 7 |
| **合计（有重叠）** | **14,784** | **216** | **1,387** | **562** |

---

## 流水线设计

```
阶段 1  DATA MINING
        关键词 × 地区切片抓取 → RegEx 从自由文本抽技能与经验 → 校验薪资串
        产出：不可变原始 JSON 快照

阶段 2  DATA PROCESSING
        薪资归一化 → 去重 → 缺陷修复策略 → 特征表 → 质量门禁
        产出：Parquet 分析表 + 数据质量报告

阶段 3  PREDICTIVE ANALYTICS
        M0 基线（地区×学历中位数）
        M1 Ridge（one-hot + TF-IDF）
        M2 LightGBM
        M3 小型 MLP
        产出：模型对比表 + 误差分析

阶段 4  证明数据与结论正确
        参数化单元测试 → 集成测试 → schema 校验 → 跨字段一致性证明
        → 泄漏审查 → 基线对比 → 稳健性分析
        产出：测试套件 + 验证报告
```

**阶段 3 的核心是 M0→M3 的逐级对比**，它回答子问题 3，而不是假定「神经网络一定更好」。

---

## 项目现状

已完成：数据源验证与 API 行为逆向 → **Stage 1 全量采集** → **Stage 2 数据加工** → **Stage 3 预测分析** → **Stage 4 验证**。

### Stage 4 验证结果

**103 项测试，约 16 秒，不需要网络**，每次 push 自动跑（见徽标）：

| 测试文件 | 用例 | 证明什么 |
|---|---:|---|
| `tests/test_textmining.py` | 43 | 解析规则读俄语广告的方式与人类一致 |
| `tests/test_repair_policy.py` | 15 | 两个数据缺陷的修复策略按声明执行 |
| `tests/test_leakage.py` | 12 | 没有任何目标派生列进入模型 |
| `tests/test_pipeline_integration.py` | 24 | 冻结 fixture 端到端 + 真实语料的不变量 |
| `tests/test_docs_links.py` | 9 | 每份文档里的每个相对链接都能解析 |

**测试抓出两个真 bug**：`parse_salary_text("до 50000")` **把上下界弄反了**（把"最高 5 万"读成了"最低 5 万"），以及 `"Java Script"`（带空格）**两个模式都不匹配**。两者都不影响已发布的结论 —— 语料里 100% 是 `"от N"`，但它们是潜伏的正确性缺陷。

**泄漏审计抓出三处文档矛盾**：2 列（`region_code`、`currency`）**从未被分类**；3 列（`text_blob`、`qualification`、`typical_position`）**同时被声明为特征和排除项**。

**测试还逼出一个更好的接口**：`build_table()` 原本不存在 —— 表格的最终形态只在 `main()` 里产生，而 `main()` 还负责写文件，所以测试拿不到「Stage 3 实际看到的那张表」。

详情见 [`docs/validation_report.md`](docs/validation_report.md)，其中也**明确列出了这个阶段没有证明的东西**（采集器无测试、指标值未被测试钉住、fixture 只有 6 条）。

### Stage 3 建模结果

`train_models.py` 在**时序留出**上对比六组模型（训练用 2026-08-20 之前的 14,513 条，测试用之后的 4,886 条）：

| 模型 | MAE (RUB) | 占中位薪资 | R² (log) |
|---|---:|---:|---:|
| M0a 全局训练中位数 | 28,594 | 57.2% | −0.177 |
| M0b 地区×学历格中位数 | 23,674 | 47.3% | 0.187 |
| M1 Ridge（稀疏 one-hot + 完整 TF-IDF） | 17,966 | 35.9% | 0.592 |
| M1b Ridge（稠密设计） | 18,264 | 36.5% | 0.588 |
| **M2 梯度提升** | **17,478** | **35.0%** | **0.615** |
| M3 神经网络（PyTorch / RTX 3080） | 17,945 | 35.9% | 0.595 |

**结论：神经网络没有跑赢梯度提升**（R² 0.595 vs 0.615，MAE 高 2.7%）。M2 与 M3 输入矩阵**逐字节相同**，所以这是纯粹的模型类别对比，不是特征对比 —— 这正是研究子问题 3 的答案，也是本项目的核心发现之一。

单个随机种子不算证据，所以 M3 又用 **7 个种子**重训了一遍：R² = 0.6002 ± 0.0038（0.595–0.605），**M2 在全部 7 个种子上、两个指标上都胜出**。诚实的表述是：梯度提升赢的幅度大致等于神经网络自身的种子噪声宽度。

比最强基线（M0b）改善 **26.2%**。产出 [`docs/model_report.md`](docs/model_report.md) 与两张图。

**控制变量后的 IT 溢价：+1.0%**。`is_it` 放进 Ridge 与其他特征一起回归，系数换算成年化薪资差异只有 1% —— 和 Stage 2 里"IT 中位数比对照组高 5,000 卢布"的**原始差距**对比鲜明：那个差距几乎完全由地区、学历、职业结构解释掉了。

### Stage 2 加工结果

`build_features.py` 把原始记录变成一张可建模的分析表：

| | 数值 |
|---|---:|
| 原始记录 | 22,387 |
| 去掉重复（同一 id / 同雇主+同岗位+同薪资） | −446 / −2,363 |
| **分析表行数** | **19,578** |
| 其中 IT / 对照组 | 11,945 / 7,633 |
| 分析表列数 | 73 |
| **质量门禁** | **10/10 通过** |

产出 `data/processed/vacancies.parquet`（2.9 MB）与 [`docs/data_quality_report.md`](docs/data_quality_report.md)。

**Stage 2 挖出两个会静默毁掉模型的陷阱：**

1. **零值哨兵**。门户用字面值 `0` 表示"薪资未指定"（原文 `"от 0"`），179 条记录如此。`log(0) = -inf`，模型会认真地从"0 卢布月薪"里学习。已置为缺失并保留标记。
2. **`requirement.experience` 不是年数，是编码**。取值 `0`/`1`/`3` 占 91%，还有 `18`、`20`、`31`、`42`。`code=0` 的岗位文本里写着"最高 35 年经验"，`code=10` 的写着"от 10 лет"而 `code=18` 的写着"от 1 года" —— 无一致解读。**该字段已从特征集中撤除**，详情见质量报告第 5 节。

### Stage 1 采集结果

`collect.py` 用 8 并发 worker，2,002 次请求、24.7 分钟完成：

| | IT 岗位 | 非 IT 对照组 |
|---|---:|---:|
| 独立岗位 | **12,656** | **9,731** |
| 覆盖地区数 | 89 | 87 |
| `salary_min` 有值 | 99% | 99% |
| `creation-date` 跨度 | 2016-07 … 2026-09 | 2015-08 … 2026-09 |

合计 **22,387 条**，压缩后 13.4 MB 入库。字段结构与缺陷统计见 [`docs/data_dictionary.md`](docs/data_dictionary.md) —— 全部由脚本从数据算出，没有一个数字是手写的。

两组数据有 **446 条重叠**（同时命中 IT 和对照组关键词），已在 Stage 2 折叠为 IT 组，保证两组互斥 —— 否则"IT 溢价"会变成部分自己和自己比。

地区目录 [`data/raw/regions.json`](data/raw/regions.json) 记录实测发现的 **78 个地区**（由探测 1–92 号编码得出，不是硬编码的清单）。

### 数据源验证（早期阶段）

[`stage0_source_verification/verify_sources.py`](stage0_source_verification/verify_sources.py) 对线上 API 跑了 8 项检查，全部通过，输出见 [`logs/verification_log.txt`](logs/verification_log.txt)。

| 检查项 | 结果 |
|---|---|
| 连通性与全国总量 | HTTP 200，**522,303** 岗位，无需认证 |
| 地区过滤验证 | `region_code` 59/77/66 分别返回彼尔姆/莫斯科/斯维尔德洛夫斯克 |
| `salary` 字段覆盖率 | **100%** |
| RegEx 薪资解析 vs `salary_min` | **100% 一致** |
| RegEx 技能抽取 | 1С / python / sql / REST / Excel / Linux / C++ 等 22 类 |
| `creation-date` 跨度 | **2020-08 … 2026-09** |

> 上表的数字是那次运行测得的。`logs/verification_log.txt` 每次运行都会被覆盖，而 Trudvsem 的实时数据库总量会变动（重新运行时全国总量显示 494,808 而非 522,303），所以日志里的绝对值与上表不同是正常的 —— 要比较的是结论，不是瞬时计数。

---

## 已知的数据质量问题

这些都是实测发现，不是猜测。它们是阶段 4 的真实素材。

### 1. `salary_min == salary_max`，占 36%

门户对 `от 40000`（"40000 起"）这类开放式广告把**同一个值写进上下界**。所以 `salary_max` 看起来有值，实际不含任何信息量。任何把 `salary_max` 当作上界使用的模型都是错的。

在 21,941 条有薪资的记录中实测到 **7,654 条（36%）** 上下界相等。

→ 处理方式：只对 `salary_min` 建模；量化该缺陷、选定并论证修复规则、报告该选择对结果的影响。

### 2. `skills` 字段 77% 为空

本该是薪资预测最强因子的技术技能，**无法直接读取**，必须从俄语自由文本 `requirements` / `duty` 中挖掘。这是 RegEx 的核心工作，不是装饰。

实测：22,387 条中 **16,978 条（77%）** 的 `skills` 为空。

### 3. 西里尔字母陷阱：43% 的 C++ 岗位被朴素正则漏掉

俄语广告经常用**西里尔字母 С**（U+0421）写 `С++`，因为它和拉丁 C 长得完全一样。Python 的 `re.IGNORECASE` **不会**跨字母表折叠：

```python
re.search(r"c\+\+", "C++", re.I)   # -> match
re.search(r"c\+\+", "С++", re.I)   # -> None   静默漏掉
```

在全部 22,387 条上实测：139 条只用拉丁写法，**102 条只用西里尔写法**，另有 100 条两种混用 —— 朴素正则的漏检率是 **102/239 ≈ 43%**。（早期 317 条样本上测得 31%，样本变大后比例更高。）

这类缺陷**不报错**，只会静默地把特征变成 0。只有看真实数据才能发现，读文档永远发现不了。这正是阶段 4 参数化单元测试存在的理由。

### 4. `salary` 自由文本格式高度单一

21,941 条有薪资的记录中，**100% 是 `"от N"` 形式**，货币 **100% 是 `«руб.»`**。所以薪资串解析是**校验工具**而非特征来源 —— 这一点反直觉，但实测如此。

---

## API 注意事项（重跑必读）

这个 API 有几个**静默失败**模式，已在 `stage0_source_verification/verify_sources.py` 的 docstring 中完整记录：

### 未知参数名被静默忽略，不报错

`regionCode`、`regionId`、`area`、`regionName` 全部返回**全国数据**并给 HTTP 200，没有任何警告。正确的名字是 `region_code`（下划线）。

> ⚠️ 不校验返回 `region.name` 的客户端，会在「以为筛了地区」的情况下拿全国数据训练模型。
> `stage0_source_verification/verify_sources.py` 对每次请求都断言返回的地区名。

### 分页已失效（行为在实测期间发生了变化）

早期实测中 `offset` 最深可用到 999。**现在 `offset > 0` 一律返回 HTTP 200 但零条记录**，同一会话内行为变了。只有 `offset=0` 可靠。

| limit | offset=0 | offset>0 |
|---|---|---|
| 100 | OK，返回 100 条 | **返回 0 条** |
| 10 | OK | **返回 0 条** |

`limit` 超过 100 会被静默截断为 100。

所以采集策略是：**每个（关键词 × 地区）单元格只发一次 `limit=100&offset=0`，覆盖率完全靠切片扩展**。单元格超过 100 条时只取前 100 条 —— 这是被接受并明确记录的损失。

### 服务端延迟 5–6 秒，必须并发

单次请求耗时 5–6 秒，且**连接池无帮助**（说明瓶颈是服务端处理，不是握手开销）。实测并发效果：

| worker 数 | 吞吐 | 失败数 |
|---|---|---|
| 顺序 | 0.19 req/s | 0 |
| 4 | 0.56 req/s | 0 |
| 8 | **1.20 req/s** | 0 |

`collect.py` 因此使用 8 个 worker —— 这是把采集从 8 小时压到半小时以内的唯一办法。

### 没有可用的日期过滤

`date_from`、`dateFrom`、`date`、`from` 被忽略；`modifiedFrom` / `modifiedTo` 直接返回 500。

但每条记录自带 `creation-date`，实测样本跨度 2020-08 到 2026-09 —— **时间轴在数据里是现成的**，只是不能作为查询条件。

---

## 排查陷阱（重跑前必读）

### `TruncatedSVD` 会在默认线程设置下挂死

`train_models.py` 里的 SVD 在 **48 个以上分量时永不返回** —— CPU 满负载、无输出、无报错、不退出。矩阵只有 14513×3000、128 万非零元，这点规模不该有问题：这是 **BLAS 线程超额订阅**，不是计算量大。

修复方式是**在 `import numpy` 之前**固定 BLAS 线程数，`train_models.py` 开头已经这样做了：

| 线程设置 | SVD(128) |
|---|---|
| 默认（全部核心） | **挂死** |
| `OPENBLAS_NUM_THREADS=1` + `MKL_NUM_THREADS=1` | **0.6 秒** |

单线程反而更快。只固定 BLAS 变量、不动 `OMP_NUM_THREADS`，所以 `HistGradientBoosting` 仍能用满所有核心（400 轮 2.5 秒）。

**如果你要自己写涉及 `TruncatedSVD` / `PCA` / 大型矩阵分解的代码，必须在导入 numpy 前加这两行：**

```python
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np   # 必须在这之后
```

### 挂死的脚本会留下占满 CPU 的孤儿进程

终端或工具超时**只杀掉外层包装进程，Python 子进程会活下来**并继续满负载空转。本项目开发期间因此积累了 3 个孤儿进程，合计烧掉约 **7.9 小时核心时间**，而且当时所有后台任务都显示"已完成"，完全没有告警。

判断方法：某个脚本长时间无输出且风扇狂转时，先查进程。

```powershell
Get-Process python                        # 看有没有残留
Get-Process python | Stop-Process -Force  # 清理
```

诊断顺序：**先看有没有孤儿进程，再怀疑代码慢。** 本项目最初把 20 分钟无输出误判为算法问题，实际是一个僵尸进程在抢 CPU。

---

## 目录结构

目录按**管道阶段**组织：属于哪个阶段的文件就放在哪个阶段文件夹里，多个阶段共用的放 `common/`。

```
it-salary-ru/
├── README.md / README.en.md / requirements.txt / LICENSE
├── .github/workflows/tests.yml     CI：双 Python 版本跑测试 + Stage 1–3
│
├── common/                         跨阶段共用
│   ├── paths.py                        全部路径的唯一来源
│   └── textmining.py                   技能 / 薪资 / 经验正则（Stage 2 与 3 共用）
│
├── stage0_source_verification/     Stage 0 —— 这个数据源到底能不能用？
│   ├── verify_sources.py               8 项数据源检查（原来是 feasibility_check.py）
│   └── probes/                         API 行为逆向的 8 个探测脚本
│       ├── probe_structure.py             记录字段结构
│       ├── probe_params.py                参数名（如何发现 region_code）
│       ├── probe_paging.py                分页参数
│       ├── probe_limits.py                limit/offset 组合极限
│       ├── probe_paging_boundary.py       offset×limit 假设的证伪
│       ├── probe_latency.py               延迟归因（服务端 vs 握手）
│       ├── probe_throttle.py              深度 offset 失效与并发验证
│       └── probe_concurrency.py           并发扩展性（4 vs 8 worker）
│
├── stage1_collection/              Stage 1 —— 采集原始数据
│   ├── collect.py                      8 并发 worker
│   ├── make_data_dictionary.py         从原始数据生成数据字典
│   └── probe_textmining_edges.py       解析边缘情况探测（证据）
│
├── stage2_processing/              Stage 2 —— 原始记录 → 可建模的表
│   └── build_features.py               清洗、修复、去重、质量门禁
│
├── stage3_analytics/               Stage 3 —— 建模与对比
│   ├── train_models.py                 M0→M3 对比、误差分析、泄漏审查
│   └── probe_seed_robustness.py       七种子稳健性探测（证据）
│
├── tests/                          Stage 4 —— 验证套件（103 项，无需网络）
│   ├── test_textmining.py                  43 项：解析规则
│   ├── test_repair_policy.py               15 项：修复策略
│   ├── test_leakage.py                     12 项：泄漏不变量
│   ├── test_pipeline_integration.py        24 项：端到端 + 语料不变量
│   ├── test_docs_links.py                   9 项：文档相对链接完整性
│   └── fixtures/raw_sample.json            6 条手工构造的冻结输入
│
├── data/
│   ├── raw/                        入库的不可变输入（~13 MB）
│   │   ├── regions.json                78 个地区目录（实测得出，非硬编码）
│   │   ├── sample_3_records.json       3 条完整记录（人类可读）
│   │   ├── trudvsem_it_harvest.json.gz     IT 岗位采集结果
│   │   └── trudvsem_control_harvest.json.gz 非 IT 对照组
│   └── processed/                  派生产物，不入库，可重新生成
│       └── vacancies.parquet           分析表（19,578 行 × 72 列）
│
├── docs/                           给读者看的报告
│   ├── final_report.md                 最终报告（英文，汇总全部阶段）
│   ├── final_report.zh.md              最终报告（中文）
│   ├── data_dictionary.md              数据字典（由脚本从数据算出，非手写）
│   ├── data_quality_report.md          Stage 2 质量门禁与修复影响
│   ├── model_report.md                 Stage 3 模型对比、泄漏审查、误差分析
│   ├── validation_report.md            Stage 4 验证报告（含未覆盖项）
│   └── figures/                        Stage 3 图表
│
└── logs/                           每次运行的输出
    ├── verification_log.txt
    └── collection_log.txt
```

`stage0_source_verification/probes/` 不是草稿 —— 它是「我怎么知道 `regionCode` 不行」「我怎么知道分页失效了」的可复现证据。各阶段文件夹里的 `probe_*.py` 同理。

**路径的唯一来源是 [`common/paths.py`](common/paths.py)。** 其他任何地方都不应该用 `__file__` 拼路径 —— 这次重构之前，五个脚本各自算路径，所以搬一个目录要改五处代码加 CI、gitignore 和文档。

---

## 如何运行

**所有命令都从仓库根目录运行**，用相对路径调用脚本。每个脚本会自己把仓库根加入 `sys.path`，所以从别处调用也可以。

```powershell
cd C:\Users\Neko\Desktop\Workspace\it-salary-ru
pip install -r requirements.txt

# 日常用到的三个（数据已在库中，无需重新采集）
python stage2_processing/build_features.py            # Stage 2 → 分析表 + 质量报告（约 30 秒）
python stage3_analytics/train_models.py               # Stage 3 → 模型报告（约 45 秒）
python -m pytest tests -v                             # Stage 4 验证（103 项，约 16 秒）

# 按需运行
python stage1_collection/make_data_dictionary.py      # 重新生成数据字典
python stage1_collection/collect.py                   # Stage 1 采集（约 25 分钟，会访问线上 API）
python stage0_source_verification/verify_sources.py   # 8 项数据源检查（会访问线上 API）
```

`collect.py` 支持参数：`--budget N`（请求上限）、`--seconds N`（时间上限）、`--phase national|grid|both`（只跑全国扫描或地区网格）、`--refresh-regions`（重新探测地区目录）。中断后区域目录会保留，下次运行自动续跑。

依赖见 [`requirements.txt`](requirements.txt)。`torch` 是**可选**的 GPU 路径 —— 缺失时 Stage 3 自动回退到 sklearn 的 MLP，测试套件也不依赖它。

环境：Anaconda Python 3.14.6，位置 `C:\ProgramData\anaconda3\python.exe`。

---

## 路线图

| 阶段 | 里程碑 | 产出 |
|---|---|---|
| ✅ 已完成 | 数据源验证与 API 逆向 | 验证脚本、验证日志、原始样本 |
| ✅ 已完成 | **Stage 1 全量采集** | 22,387 条原始数据、78 地区目录、数据字典 |
| ✅ 已完成 | **Stage 2 清洗、RegEx 抽取、质量门禁** | 19,578 行分析表、10/10 门禁报告 |
| ✅ 已完成 | **Stage 3 预测分析（M0→M3 对比）** | 模型报告、误差分析、两张图 |
| ✅ 已完成 | **Stage 4 单元测试、集成测试、泄漏审查** | 103 项测试、CI 工作流、验证报告 |
| ✅ 已完成 | **成文** | 最终报告（[英](docs/final_report.md) / [中](docs/final_report.zh.md)）、MIT 许可 |

---

## 数据源的已知局限

需要如实说明：门户偏向国企和蓝领岗位，IT 仅占约 15k / 522k。这是数据源的客观局限，结论只在该人群内有效。IT 岗位分析以全俄为主体，彼尔姆作为子分析呈现。

---

## 许可

**代码**（`common/`、`stage*/`、`tests/`、文档与图表）采用 [MIT 许可](LICENSE) —— 任何人可自由使用、修改、再分发。

**`data/raw/` 下的岗位记录不在此授权范围内。** 它们是 Трудвсем（«Работа России»，由 Роструд 运营）发布的官方开放数据，仍受其发布方条款约束：

- 数据来源：https://opendata.trudvsem.ru/api/v1/vacancies
- 条款：https://trudvsem.ru/opendata/api

以 gzip 形式随仓库分发**仅为使分析可复现**，无需重新采集。复用者应查阅上述条款并引用门户为数据来源；也可以随时用 `stage1_collection/collect.py` 直接从 API 重新生成等价数据集。
