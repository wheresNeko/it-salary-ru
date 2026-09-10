# 俄罗斯 IT 岗位薪资预测

基于俄罗斯国家就业门户开放数据的机器学习项目。目标是从招聘信息中预测 IT 岗位的薪资水平，并量化「哪些因素真正决定薪资」。

> **English version: [README.en.md](README.en.md)**

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

**已完成**：数据源选择与实测验证、API 行为逆向、流水线设计。

`feasibility_check.py` 对线上 API 跑了 8 项检查，全部通过。完整输出见 [`verification_log.txt`](verification_log.txt)。

| 检查项 | 结果 |
|---|---|
| 连通性与全国总量 | HTTP 200，**522,303** 岗位，无需认证 |
| 实际抓取 | **317 条独立岗位**，56 次请求，**0 失败** |
| 地区过滤验证 | `region_code` 59/77/66 分别返回彼尔姆边疆区/莫斯科市/斯维尔德洛夫斯克州 |
| `salary` 字段覆盖率 | **100%** |
| `salary_min` / `salary_max` | 97% / 91% |
| RegEx 薪资解析 vs `salary_min` | **100% 一致** |
| RegEx 技能抽取 | 1С=134, python=52, sql=35, REST=30, Excel=29, Linux=26, git=19, C++=18 |
| 从自由文本抽取经验要求 | 1年×10, 2年×3, 3年×7, 5年×4, 7年×7 |
| `creation-date` 跨度 | **2020-08 … 2026-09** |

---

## 已知的数据质量问题

这些都是实测发现，不是猜测。它们是阶段 4 的真实素材。

### 1. `salary_min == salary_max`，占 34%

门户对 `от 40000`（"40000 起"）这类开放式广告把**同一个值写进上下界**。所以 `salary_max` 看起来有值，实际不含任何信息量。任何把 `salary_max` 当作上界使用的模型都是错的。

→ 处理方式：只对 `salary_min` 建模；量化该缺陷、选定并论证修复规则、报告该选择对结果的影响。

### 2. `skills` 字段 81% 为空

本该是薪资预测最强因子的技术技能，**无法直接读取**，必须从俄语自由文本 `requirements` / `duty` 中挖掘。这是 RegEx 的核心工作，不是装饰。

### 3. 西里尔字母陷阱：31% 的 C++ 岗位被朴素正则漏掉

俄语广告经常用**西里尔字母 С**（U+0421）写 `С++`，因为它和拉丁 C 长得完全一样。Python 的 `re.IGNORECASE` **不会**跨字母表折叠：

```python
re.search(r"c\+\+", "C++", re.I)   # -> match
re.search(r"c\+\+", "С++", re.I)   # -> None   静默漏掉
```

实测：26 个 C++ 岗位中，**8 个（31%）只用西里尔写法**，朴素正则完全匹配不到 —— 其中包括一个岗位标题写着「Ведущий программист **С++**(Qt)」。

这类缺陷**不报错**，只会静默地把特征变成 0。只有看真实数据才能发现，读文档永远发现不了。这正是阶段 4 参数化单元测试存在的理由。

### 4. `salary` 自由文本格式高度单一

样本中 100% 是 `"от N"` 形式，货币 100% 是 `«руб.»`。所以薪资串解析是**校验工具**而非特征来源 —— 这一点反直觉，但实测如此。

---

## API 注意事项（重跑必读）

这个 API 有几个**静默失败**模式，已在 `feasibility_check.py` 的 docstring 中完整记录：

### 未知参数名被静默忽略，不报错

`regionCode`、`regionId`、`area`、`regionName` 全部返回**全国数据**并给 HTTP 200，没有任何警告。正确的名字是 `region_code`（下划线）。

> ⚠️ 不校验返回 `region.name` 的客户端，会在「以为筛了地区」的情况下拿全国数据训练模型。
> `feasibility_check.py` 对每次请求都断言返回的地区名。

### 分页只有一种可靠组合

实测矩阵（OK = HTTP 200，X = HTTP 500）：

| limit | off=0 | off=10 | off=100 | off=500 | off=900 | off=999 |
|---|---|---|---|---|---|---|
| 10 | OK | OK | OK | OK | OK | OK |
| 20 | OK | OK | OK | X | X | X |
| 50 | OK | OK | OK | X | X | X |
| 100 | OK | OK | OK | X | X | X |

只有 `limit=10` + offset 步进在任意深度都可用。最大 offset 是 **999**，所以单个（关键词 × 地区）查询最多约 1000 条 —— 覆盖率靠**切片查询**扩展，不靠深分页。

### 没有可用的日期过滤

`date_from`、`dateFrom`、`date`、`from` 被忽略；`modifiedFrom` / `modifiedTo` 直接返回 500。

但每条记录自带 `creation-date`，实测样本跨度 2020-08 到 2026-09 —— **时间轴在数据里是现成的**，只是不能作为查询条件。

---

## 目录结构

```
it-salary-ru/
├── README.md                    本文件（中文）
├── README.en.md                 English version
├── feasibility_check.py         数据源验证脚本（已跑通，8 项检查）
├── verification_log.txt         验证脚本的完整输出（运行后生成）
├── raw_samples/                 真实抓取的数据
│   ├── sample_3_records.json        3 条完整记录（人类可读，用于展示字段结构）
│   └── trudvsem_raw_harvest.json    317 条岗位（1.4 MB）
└── api_investigation/           API 行为逆向过程的探测脚本
    ├── probe_structure.py           记录字段结构
    ├── probe_params.py              参数名（找出 region_code）
    ├── probe_paging.py              分页参数与 offset 上限
    └── probe_limits.py              limit/offset 组合极限
```

`api_investigation/` 不是草稿 —— 它是「我怎么知道 `regionCode` 不行」的可复现证据。

---

## 如何运行

```powershell
cd C:\Users\Neko\Desktop\Workspace\it-salary-ru
pip install requests
python feasibility_check.py
```

脚本会访问线上 API、抓取样本、重写 `verification_log.txt` 和 `raw_samples/`。

依赖：`requests`（必需）。DOM 解析阶段还需要 `pip install beautifulsoup4 lxml`。

环境：Anaconda Python 3.14.6，位置 `C:\ProgramData\anaconda3\python.exe`。

---

## 路线图

| 阶段 | 里程碑 | 产出 |
|---|---|---|
| ✅ 已完成 | 数据源验证与 API 逆向 | 验证脚本、验证日志、原始样本 |
| 下一步 | 全量采集（全地区 × 全关键词 + 非 IT 对照组） | 原始数据集 + 数据字典 |
| | 清洗、RegEx 抽取、质量门禁 | 分析表 + 数据质量报告 |
| | 探索性分析、特征工程 | EDA notebook、特征规范 |
| | M0 基线与 M1 Ridge | 评估框架、首批真实指标 |
| | M2 LightGBM 与 M3 MLP 对比 | 模型对比表、误差分析 |
| | 单元测试、集成测试、泄漏审查 | 测试套件、验证报告 |
| | 成文与展示 | 最终报告、可复现仓库 |

---

## 数据源的已知局限

需要如实说明：门户偏向国企和蓝领岗位，IT 仅占约 15k / 522k。这是数据源的客观局限，结论只在该人群内有效。IT 岗位分析将以全俄为主体，彼尔姆作为子分析呈现。
