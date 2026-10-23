# DT-MemEval

双时间轴会话记忆数据集与评估项目，数据包含 **4 个领域序列、519 个会话、2,000 道问题、8 类任务**。

## 安装

推荐 Python 3.11。不同记忆基线的依赖可能冲突，请分别创建环境。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

安装后可直接运行下方三个实验入口，数据加载与协议校验由运行代码自动完成。

## 最小完整流程：Long-Context

从项目根目录准备独立的回答模型及裁判配置，填写真实服务商支持的模型名称、API key 和地址：

```bash
cp configs/qa_api.example.json configs/qa_api.local.json
cp configs/judge_api.example.json configs/judge_api.local.json
```

构建记忆：

```bash
python memory_construction.py \
  --memory-type Long-Context --dataset-type dualtimeline \
  --dataset-path data/standard.json \
  --config-path configs/dualtimeline_long_context.json
```

检索：

```bash
python memory_search.py \
  --memory-type Long-Context --dataset-type dualtimeline \
  --dataset-path data/standard.json \
  --config-path configs/dualtimeline_long_context.json --top-k 10
```

回答并评分：

```bash
python memory_evaluation.py \
  --dataset-type dualtimeline \
  --search-results-path memory/dualtimeline_long_context/10_0_4.json \
  --qa-model gpt-4o-mini --api-config-path configs/qa_api.local.json \
  --judge-model gpt-4o-mini --judge-api-config-path configs/judge_api.local.json \
  --judge-batch-size 4 \
  --prompt-template membase.datasets.dualtimeline.qa_prompt --metrics llm_judge
```

记忆构建和检索按完整领域序列运行；全部会话写入后统一回答。D1/D2 题目中的历史截止时间继续约束答案。

## 记忆基线

模型路径为公开模型标识，需要时将公开标识改为服务器上预下载的本地模型路径。

| `memory-type` | 配置 | 额外依赖 |
|---|---|---|
| `Long-Context` | `configs/dualtimeline_long_context.json` | 核心依赖 |
| `A-MEM` | `configs/dualtimeline_amem.json` | `pip install -r envs/amem_requirements.txt` |
| `LangMem` | `configs/dualtimeline_langmem.json` | `pip install -r envs/langmem_requirements.txt` |
| `Mem0` | `configs/dualtimeline_mem0.json` | `pip install -r envs/mem0_requirements.txt` |
| `NaiveRAG` | `configs/dualtimeline_rag.json` | `pip install -r envs/rag_requirements.txt` |
| `LightMem` | `configs/dualtimeline_lightmem_session_fixed.json` | `pip install -r envs/lightmem_requirements.txt` |

基线环境建议先安装对应 `envs` 依赖，再 `pip install -e .`。

## 评分输出

`--metrics llm_judge` 产生三个指标：

- `llm_judge`：核心答案是否正确。
- `evidence_judge`：引用的会话日期是否充分且归属正确。
- `grounded_accuracy`：答案与证据同时正确。

输出包括逐题结果及 overall / by_task / by_dimension 汇总。裁判调用失败计为 error，值为 null；准确率分母采用成功评分条数。回答在裁判调用前单独保存。

## 数据与提示词

- `data/standard.json`：会话、问题、公开提示词及裁判专用答案/证据。
- 回答模型接收 `metadata.system_prompt` 作为 system 消息，user 消息由检索上下文、该题 `user_prompt` 与 `QUESTION: {text}` 组成。
- `evaluation`、标准答案、证据审计和构造查询仅供评估，不能输入记忆系统或回答模型。

## 目录与来源

```text
membase/                 MemBase 的运行代码与基线
memory_construction.py   记忆构建入口
memory_search.py         检索入口
memory_evaluation.py     回答与评分入口
data/       实验数据集
configs/                 示例配置
envs/                    各基线的安装依赖
```

项目名称为 **DT-MemEval**，内部保留原框架包名 `membase`。这是基于 MemBase 的派生项目，来源和版权见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)、[LICENSE-MemBase](LICENSE-MemBase)。



MemBase 采用 MIT 许可证，允许使用、修改和分发；随代码保留其版权声明及完整许可声明，见 `LICENSE-MemBase`。第三方基线保留各自适用的许可证。
