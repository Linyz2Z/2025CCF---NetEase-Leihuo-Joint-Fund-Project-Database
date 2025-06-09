# RouterJSD技术方案

## 1. 项目概述

在多模型问答系统中，不同大语言模型在不同任务上的表现存在差异。为了在保证准确率的前提下控制计算成本，我们设计了一个模型路由器`RouterJSD`，旨在无需调用所有大语言模型的情况下，根据查询内容动态选择最佳的大语言模型。

## 2. 问题定义

- **输入**：
  - `question`：中文或英文自然语言问题

- **输出**：
  - `model name`：候选 LLM 中预测该问题表现最优的模型名称（如 gpt_4o, qwen25_72b_instruct）

## 3. 技术路线

### 3.1 查询嵌入表示
- 使用轻量多语言语义编码器 `mDeBERTa-v3-base`
- 输出维度为 768，用于表示问题的语义表示

### 3.2 LLM嵌入表示
- 初始化为可学习参数
- 每个候选 LLM 对应一个 768 维向量，表示其能力特征

### 3.3 相似度计算
- 使用余弦相似度计算查询向量与每个 LLM 嵌入向量的相似度
  
### 3.4 损失函数
-  使用 JS 散度（Jensen-Shannon Divergence）作为训练目标
-  优化查询嵌入向量与各候选 LLM 嵌入向量之间的匹配程度

### 3.5 训练策略
- 基于标注数据进行监督训练
- 同时优化编码器与LLM嵌入向量

## 4. 模块设计说明
```
project/ 
├── code/                   # 项目源代码
│ ├── configs/              # 模型配置文件
│ ├── test_data/            # 测试数据
│ ├── test_result/          # 测试结果
│ ├── router_dataset.py     # 数据集定义
│ ├── router_model.py       # 模型结构定义
│ └── test.py               # 模型推理程序
├── demo/                   # 演示程序与脚本
├── model/                  # 模型权重文件
├── docs/                   # 项目相关文档
│ └── design.md             # 技术方案说明文档
├── requirements.txt        # 依赖库
└── README.md               # 项目说明文档
```

## 5. 数据格式示例

- **训练集**

| id | question | qwen25_72b_instruct | gpt_4o_mini_cot | ministral_8b_instruct_2410 | gpt_4o_mini_cot | ... | llama31_405b_instruct |
|---|-----|-|-|-|-|-|-|
| 0 | question | 0/1 | 0/1 | 0/1 | 0/1 | ... | 0/1 | 

- **测试集**

| id | question | pred |
|---|-----|-------|
| 0 | question | model_name |

## 6. 候选模型列表

```
"qwen25_72b_instruct",
"gpt_4o_mini_cot",
"ministral_8b_instruct_2410",
"deepseek_chat",
"glm_4_plus",
"llama31_8b_instruct",
"qwen25_32b_int4",
"gpt_4o",
"glm_4_air",
"gpt_4o_mini",
"qwen25_math_7b_instruct",
"llama31_70b_instruct",
"mistral_7b_instruct_v02",
"mixtral_8x7b_instruct",
"glm_4_flash",
"qwq_32b_preview",
"gemini15_flash",
"deepseek_coder",
"qwen25_7b_instruct",
"llama31_405b_instruct"
```

## 7. 运行方式

```bash
bash demo/test.sh
```

预测结果保存在 [`code/test_results/`](.code/test_results/) 文件夹下的 CSV 文件中。

## 8. 评价指标

- 准确率（Accuracy）：预测模型是否与真实表现好的模型一致

## 9. 示例结果
[`code/test_results/demo_test_pred.csv`](./code/test_data/demo_test_pred.csv)

| id | question | pred |
|---|-----|-------|
| 0 | 关于：poetry_sentiment_analysis <br> 问题：古诗词“一麾来此恰三年\|到得终更分外难\|老眼看灯浑作晕\|愁心得酒不成欢“的整体情感是____<br>选项：<br> A. 消极的 <br> B. 无法判断 <br> C. 中性的 <br> D. 积极的  | gpt_4o |
| 1 | 关于：basic_ancient_chinese <br> 问题：按照传统的“六书”体例，“吹”字应属 <br> 选项：<br> A. 指事 <br> B. 象形 <br> C. 会意 <br> D. 形声  | qwen25_72b_instruct |
| 2 | 关于：couplet_prediction <br> 问题：“一定精神空世界”的下联最可能是___。<br> 选项：<br> A. 十分恩爱老夫妻 <br> B. 叩头北阙作庸臣 <br> C. 子夜无心月入怀 <br> D. 我学草字有原因  | deepseek_chat |