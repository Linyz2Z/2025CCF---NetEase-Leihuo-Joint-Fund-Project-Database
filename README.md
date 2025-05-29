# 2025CCF-网易雷火联合基金课题数据库

code：存放项目源代码，按功能或模块组织，需安装依赖确保运行。

demo：提供样例数据和示例结果，含演示脚本与配置，可以速览项目功能。

model：存放训练好的模型文件，使用时要确保正确加载。

readme：即当前目录，项目文档说明，如运行指南、使用手册，使用前建议详读。

## RouterJSD

本项目实现了一种基于 **JS 散度（Jensen-Shannon Divergence）** 的查询路由方法，目标是在调用尽可能少的大语言模型的前提下，为每个查询智能选择最合适的模型。

该方法主要由两部分构成：

- **查询编码器**：基于轻量级语言模型架构 mDeBERTa-v3-base，用于提取输入查询的语义表示；

- **LLM嵌入向量**：对大型语言模型候选集进行可学习的向量化建模，用于表示各模型的能力特征。

训练过程中使用基于 **JS 散度** 的损失函数来优化编码器输出与各候选 LLM 嵌入之间的匹配程度，从而实现高效、准确的模型选择。

## 项目功能
RouterJSD 提供了一套智能调度机制，旨在无需调用所有大语言模型的情况下，根据查询内容动态选择最佳的大语言模型。

当用户查询输入后，智能调度算法会根据查询中的任务信息、当前系统资源状态以及预设的性能和成本目标，动态评估可用的大模型选项。通过综合考虑模型的准确性、响应速度、计算资源需求和API成本等因素，智能调度算法能够实时选择出当前最适合的大模型，以确保在满足性能要求的同时，优化计算资源的使用和成本效益。

## 快速开始

### 1. 配置环境
创建conda环境并安装依赖：
```bash
conda create -n RouterLCD python=3.11
conda activate RouterLCD

# 安装对应版本的 torch 和 torchvision
pip install torch torchvision torchaudio

pip install -r requirements.txt
```

### 2. 下载权重文件
请前往[百度网盘](https://pan.baidu.com/)下载所需的权重文件，然后将其解压至 项目根目录下的`model/` 文件夹中，并确保目录结构如下所示：

```
model/
├── aclue_jcd_seed42
│ └── best_model.pth
├── aclue_jcd_seed526
│ └── best_model.pth
└──......
```
每个子文件夹以对应的模型名称命名，内部包含一个名为 best_model.pth 的权重文件。

### 3. 运行
```bash
bash demo/test.sh
```

### 4. 结果
在 `code/test_results/` 文件夹中查看对应的csv结果文件。

## 项目结构说明

```
project/ 
├── code/                   # 项目源代码
│ ├── configs/              # 模型配置文件
│ ├── test_data/            # 测试数据
│ ├── test_result/          # 测试结果
│ ├── val_data/             # 验证数据
│ ├── router_dataset.py     # 数据集定义
│ ├── router_model.py       # 模型结构定义
│ └── test.py               # 模型推理程序
├── demo/                   # 演示程序与脚本
├── model/                  # 模型权重文件
├── requirements.txt        # 依赖库
└── README.md               # 项目说明文档
```
