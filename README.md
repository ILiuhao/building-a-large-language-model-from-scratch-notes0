# 《从零构建大模型》详细学习笔记

> **Build a Large Language Model (From Scratch)** — Sebastian Raschka
> 中文版：《从零构建大模型》（人民邮电出版社·图灵）

**本笔记的资料来源（三源交叉印证）：**

| 来源 | 说明 |
|---|---|
| 📕 **原书** | 《从零构建大模型（True PDF，官方原版）》，343 页，第 1~7 章 + 附录 A~F |
| 💻 **官方代码仓库** | [`rasbt/LLMs-from-scratch`](https://github.com/rasbt/LLMs-from-scratch)（Apache 2.0），书中全部代码 |
| 📝 **知乎读书笔记** | 《从零构建大模型》读书笔记（作者在 MacBook Pro M4 Pro 24G 上的实操记录） |

原书以 GPT-2 small（1.24 亿参数）为目标，在**消费级笔记本**上完整走通"数据处理 → 模型构建 → 预训练 → 微调"的全流程。本笔记按章节组织，**每个知识点都尽量回到原书原文与官方代码**，并补充了知乎笔记中的实机运行结果。

---

## 目录

- [0. 全局地图：三阶段路线图](#0-全局地图三阶段路线图)
- [1. 环境准备与仓库结构](#1-环境准备与仓库结构)
- [2. 第 1 章 理解大语言模型](#2-第-1-章-理解大语言模型)
- [3. 第 2 章 处理文本数据](#3-第-2-章-处理文本数据)
- [4. 第 3 章 编码注意力机制](#4-第-3-章-编码注意力机制)
- [5. 第 4 章 从头实现 GPT 模型](#5-第-4-章-从头实现-gpt-模型)
- [6. 第 5 章 在无标签数据上预训练](#6-第-5-章-在无标签数据上预训练)
- [7. 第 6 章 针对分类的微调](#7-第-6-章-针对分类的微调)
- [8. 第 7 章 通过微调遵循人类指令](#8-第-7-章-通过微调遵循人类指令)
- [9. 附录 A~F](#9-附录-af)
- [10. 超参数速查表](#10-超参数速查表)
- [11. 踩坑清单](#11-踩坑清单)
- [12. 延伸阅读](#12-延伸阅读)

---

## 0. 全局地图：三阶段路线图

原书第 1 章图 1-9 给出了贯穿全书的路线图，**9 个步骤分 3 个阶段**：

```
第一阶段：实现模型架构和准备数据
  (1) 准备文本数据           → 第 2 章
  (2) 实现注意力机制          → 第 3 章
  (3) 实现 GPT 模型架构       → 第 4 章

第二阶段：预训练大语言模型，得到基础模型
  (4) 在无标签数据上预训练     → 第 5 章
  (5) 加载 OpenAI 预训练权重   → 第 5 章 5.5 节

第三阶段：微调基础模型
  (6) 微调为文本分类器        → 第 6 章
  (7) 微调以遵循指令          → 第 7 章
```

> 原书原文："在第一阶段，我们将学习数据预处理的基本流程，并着手实现大语言模型的核心组件——注意力机制。在第二阶段，我们将学习如何编写代码并预训练一个能够生成新文本的类 GPT 大语言模型……最后，在第三阶段，我们将对一个预训练后的大语言模型进行微调，使其能够执行回答查询、文本分类等任务。"

---

## 1. 环境准备与仓库结构

### 1.1 获取代码

```bash
git clone --depth 1 https://github.com/rasbt/LLMs-from-scratch.git
cd LLMs-from-scratch
```

### 1.2 创建环境（书中推荐 uv）

```bash
uv venv --python=python3.10
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
python setup/02_installing-python-libraries/python_environment_check.py
```

`requirements.txt`（带章节标注）：

```
torch >= 2.3.0          # all
jupyterlab >= 4.0       # all
tiktoken >= 0.5.1       # ch02; ch04; ch05
matplotlib >= 3.7.1     # ch04; ch06; ch07
tensorflow >= 2.18.0    # ch05; ch06; ch07
tqdm >= 4.66.1          # ch05; ch07
numpy >= 1.26, < 2.1    # dependency of several other libraries like torch and pandas
pandas >= 2.2.1         # ch06
psutil >= 5.9.5         # ch07; already installed automatically as dependency of torch
```

启动 JupyterLab：`jupyter lab` → 浏览器打开 `http://localhost:8888/lab`。

### 1.3 仓库目录结构

| 目录 | 内容 | 主代码文件 |
|---|---|---|
| `ch01` | Understanding Large Language Models | **无代码** |
| `ch02` | Working with Text Data | `ch02.ipynb`、`dataloader.ipynb`、`exercise-solutions.ipynb` |
| `ch03` | Coding Attention Mechanisms | `ch03.ipynb`、`multihead-attention.ipynb` |
| `ch04` | Implementing a GPT Model from Scratch | `ch04.ipynb`、`gpt.py` |
| `ch05` | Pretraining on Unlabeled Data | `ch05.ipynb`、`gpt_train.py`、`gpt_generate.py` |
| `ch06` | Finetuning for Text Classification | `ch06.ipynb`、`gpt_class_finetune.py` |
| `ch07` | Finetuning to Follow Instructions | `ch07.ipynb`、`gpt_instruction_finetuning.py`、`ollama_evaluate.py` |
| `appendix-A` | Introduction to PyTorch | `code-part1.ipynb`、`code-part2.ipynb`、`DDP-script.py` |
| `appendix-B` | References and Further Reading | 无代码 |
| `appendix-C` | Exercise Solutions | 习题解答清单 |
| `appendix-D` | Adding Bells and Whistles to the Training Loop | `appendix-D.ipynb` |
| `appendix-E` | Parameter-efficient Finetuning with LoRA | `appendix-E.ipynb` |

`setup/` 目录另有：`01_optional-python-setup-preferences`、`02_installing-python-libraries`、`03_optional-docker-environment`。

### 1.4 硬件要求（原书 README 原文要点）

- 主章节代码**设计为可在普通笔记本电脑上、在合理时间内运行**；
- **不需要专用硬件**；代码会自动使用 GPU（如可用）；
- 作者本人用 MacBook Air M3 / M4 全程跑通。

### 1.5 `llms_from_scratch` 包（重要）

原书后期章节把前面实现的类打包成了一个可安装的库，**避免重复粘贴代码**：

```bash
uv pip install llms_from_scratch
```

```python
from llms_from_scratch.ch02 import create_dataloader_v1
from llms_from_scratch.ch03 import MultiHeadAttention
from llms_from_scratch.ch04 import GPTModel, generate_text_simple
```

---

## 2. 第 1 章 理解大语言模型

> 本章无代码，是全书的概念地基。

### 2.1 什么是大语言模型

- 大语言模型（LLM）是一种用于**理解、生成和响应**类似人类语言文本的**神经网络**，属于深度神经网络，通过大规模文本数据训练而成。
- "大"有两层含义：**训练数据集庞大** + **模型参数规模庞大**（数百亿至数千亿个参数）。
- 核心训练任务：**下一单词预测（next-word prediction）**。它利用了语言具有顺序这一特性，让模型理解上下文、结构和关系。
- 原文的一个关键澄清：所谓模型"理解"语言，指的是**能够处理和生成看似连贯且符合语境的文本**，并不意味着它具有人类一样的意识。

### 2.2 领域层级关系

```
人工智能 (AI)
  └── 机器学习 (ML)          ← 从数据中学习，无须显式编程
        └── 深度学习 (DL)     ← 使用 3 层及以上的神经网络
              └── 大语言模型   ← 处理/生成类人语言的文本
```

传统机器学习**依赖人工特征工程**（如垃圾邮件分类需专家挑出"prize/win/free"、感叹号数量等特征）；深度学习**不需要人工提取特征**，但仍需要标签。

### 2.3 两阶段训练范式

| 阶段 | 数据 | 目标 | 产物 |
|---|---|---|---|
| **预训练** | 海量**无标注**原始文本（raw text） | 下一单词预测（**自监督**，标签由数据自身生成） | 基础模型（foundation model），如 GPT-3 |
| **微调** | 较小的**带标注**数据集 | 适应特定任务/领域 | 分类器 / 指令助手 |

两种主流微调方式：

- **指令微调**：数据集是"指令−答案"对（如"原文−正确翻译"）；
- **分类微调**：数据集是"文本−类别标签"对（如"垃圾邮件/非垃圾邮件"）。

> 原书原文："传统的机器学习模型和通过常规监督学习范式训练的深度神经网络通常需要标签信息。然而，这并不适用于大语言模型的预训练阶段。在此阶段，大语言模型使用**自监督学习**，模型从输入数据中生成自己的标签。"

### 2.4 Transformer 架构

- 2017 年 Google 论文 **"Attention Is All You Need"** 首次提出。
- 原始结构 = **编码器（encoder）+ 解码器（decoder）**，为机器翻译设计。
- 关键组件：**自注意力机制（self-attention）**，让模型衡量序列中不同词元的相对重要性，捕捉长距离依赖。

| 模型 | 使用部分 | 训练方式 | 擅长 |
|---|---|---|---|
| **BERT** | 仅编码器 | 掩码预测（masked word prediction） | 文本分类、情感分析 |
| **GPT** | 仅解码器 | 下一单词预测 | 文本生成、翻译、摘要、写代码 |

- GPT 是**自回归（autoregressive）模型**：把之前的输出作为未来预测的输入，逐词生成。
- **零样本（zero-shot）**：无任何示例即可泛化到新任务；**少样本（few-shot）**：从用户给出的少量示例中学习。
- **涌现（emergence）**：模型能完成未经明确训练的任务，如翻译——这是广泛接触多语言数据的自然结果。

> 注意区分："并非所有 Transformer 都是大语言模型（Transformer 也用于计算机视觉），也并非所有大语言模型都基于 Transformer（存在基于循环/卷积架构的 LLM）。"

### 2.5 数据集规模（表 1-1，GPT-3 预训练数据集）

| 数据集 | 描述 | 词元数量 | 占比 |
|---|---|---|---|
| CommonCrawl（过滤后） | 网络抓取数据 | 4100 亿 | 60% |
| WebText2 | 网络抓取数据 | 190 亿 | 22% |
| Books1 | 图书语料库 | 120 亿 | 8% |
| Books2 | 图书语料库 | 550 亿 | 8% |
| Wikipedia | 高质量文本 | 30 亿 | 3% |

> 表中词元数合计 4990 亿，但 **GPT-3 实际只在 3000 亿词元上训练**。CommonCrawl 的 4100 亿词元约需 **570 GB** 存储。预训练 GPT-3 的云计算成本估计高达 **460 万美元**。

### 2.6 规模对比

- 原始 Transformer：编码器、解码器各重复 **6** 次；
- GPT-3：共 **96 层** Transformer，**1750 亿**参数；
- 本书目标：GPT-2 small，**1.24 亿**参数。

### 2.7 本章小结（原书 1.8 节）

- LLM 革新了 NLP，引入了基于深度学习的新方法；
- 训练两步走：先海量无标注文本预训练（下一词预测），再小规模标注数据微调；
- 核心架构是 Transformer，核心组件是注意力机制；
- GPT 类模型只实现解码器部分，简化了架构；
- 大规模语料是预训练的关键；微调后的模型可在特定任务上超越通用模型。

---

## 3. 第 2 章 处理文本数据

> 对应代码：`ch02/01_main-chapter-code/ch02.ipynb`
> 本章完成路线图第一阶段的第 (1) 步：**实现数据采样流水线**。

### 3.1 理解词嵌入

- 深度神经网络**无法直接处理原始文本**——文本是离散的，无法做数学运算，必须转成**连续值的向量**。
- 这个过程叫**嵌入（embedding）**：把离散对象（单词、图像、文档）映射到连续向量空间中的点。
- 不同数据类型需要**不同的嵌入模型**（文本嵌入模型不适用于音频/视频）。
- 早期流行方法 **word2vec**：核心思想是"出现在相似上下文中的词往往具有相似含义"，因此语义相近的词在嵌入空间中彼此靠近（如各种鸟类的距离比国家与城市的距离更近）。
- **LLM 通常自己生成嵌入**，作为输入层的一部分在训练中更新——好处是能针对特定任务和数据优化。
- 嵌入维度从一维到数千维不等，是**性能与效率的权衡**：
  - GPT-2 small（1.17 亿）→ **768**
  - GPT-3（1750 亿）→ **12288**

### 3.2 文本分词

以 Edith Wharton 的短篇小说 **The Verdict** 为数据集（`ch02/01_main-chapter-code/the-verdict.txt`）：

```python
import urllib.request
url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
urllib.request.urlretrieve(url, "the-verdict.txt")

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))   # 20479
print(raw_text[:99])
```

**用正则逐步打磨一个简易分词器：**

```python
import re
text = "Hello, world. This, is a test."

# ① 按空白字符分割
re.split(r'(\s)', text)
# ['Hello,', ' ', 'world.', ' ', 'This,', ' ', 'is', ' ', 'a', ' ', 'test.']

# ② 在空白、逗号、句号处分割
re.split(r'([,.]|\s)', text)
# ['Hello', ',', '', ' ', 'world', '.', '', ' ', ...]

# ③ 去掉空串
[item for item in re.split(r'([,.]|\s)', text) if item.strip()]
# ['Hello', ',', 'world', '.', 'This', ',', 'is', 'a', 'test', '.']

# ④ 支持更多标点与双破折号
text = "Hello, world. Is this-- a test?"
re.split(r'([,.:;?_!"()\']|--|\s)', text)
# ['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']
```

应用到全文：

```python
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))     # 4690 个词元（不含空白）
```

> 原文提示：是否保留空白字符取决于场景。**Python 代码对缩进敏感**，若模型需保持精确结构就应保留；本书为简化暂时移除。

### 3.3 将词元转换为词元 ID

```python
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)            # 1130

vocab = {token: integer for integer, token in enumerate(all_words)}
# ('!', 0) ('"', 1) ("'", 2) ... ('Her', 49) ('Hermia', 50)
```

**代码清单 2-3：简单分词器 SimpleTokenizerV1**

```python
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)   # 去掉标点前的空格
        return text
```

**局限**：遇到训练集中没出现过的词会报错。

```python
tokenizer.encode("Hello, do you like tea?")
# KeyError: 'Hello'
```

### 3.4 引入特殊上下文词元

引入两个特殊词元：`<|unk|>`（未知词）与 `<|endoftext|>`（文本分隔）：

```python
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}
print(len(vocab.items()))     # 1132
# ...
# ('<|endoftext|>', 1130)
# ('<|unk|>', 1131)
```

**代码清单 2-4：SimpleTokenizerV2**（未知词替换为 `<|unk|>`）

```python
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    # decode 同 V1
```

验证：

```python
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
tokenizer.encode(text)
# [1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7]
#  ↑unk                                                   ↑unk
```

**其他常见特殊词元：**

| 词元 | 作用 |
|---|---|
| `[BOS]` | 序列开始，标记文本起点 |
| `[EOS]` | 序列结束，连接多个不相关文本 |
| `[PAD]` | 填充，使批次内文本等长 |

> 关键区别：**GPT 系列分词器不用这些**，只使用 `<|endoftext|>`，且它同时充当填充词元；也不用 `<|unk|>`——而是用 **BPE** 把生词拆成子词。

### 3.5 BPE（字节对编码）

- BPE 是 GPT-2、GPT-3、ChatGPT 原始模型使用的分词方案。
- 原理：**把不在词汇表中的单词分解为更小的子词单元甚至单个字符**，因此可以解析任何单词。
- 迭代构建过程：先把所有单字符加入词汇表，再把频繁共现的字符合并为子词（如 `d`+`e`→`de`），再把频繁子词合并为词，合并由**频率阈值**决定。
- 原始实现在 [openai/gpt-2/src/encoder.py](https://github.com/openai/gpt-2/blob/master/src/encoder.py)；本书用 **tiktoken**（Rust 重写核心算法，性能更高）。

```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunknownPlace.")
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
# [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250,
#  8812, 2114, 286, 617, 34680, 27271, 13]
print(tokenizer.decode(integers))
# Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.
```

两个关键观察：

1. `<|endoftext|>` 的 ID 是 **50256**，是 GPT-2 词汇表（共 **50257**）中最大的 ID；
2. **生词 `someunknownPlace` 也能正确编解码**——BPE 把它拆成了子词，无须 `<|unk|>`。

> **练习 2.1**：用 tiktoken 对 "Akwirw ier" 分词，打印所有词元 ID，再对每个 ID 单独 `decode`，重现"拆子词"的映射。

### 3.6 使用滑动窗口进行数据采样

```python
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))          # 5145

enc_sample = enc_text[50:]    # 丢掉前 50 个词元，让示例更有趣

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]
print(f"x: {x}")              # x: [290, 4920, 2241, 287]
print(f"y:      {y}")         # y:      [4920, 2241, 287, 257]
```

`y` 就是 `x` **右移一位**——这就是"下一单词预测"的输入-目标对。

**用 PyTorch Dataset + DataLoader 实现（`create_dataloader_v1`）：**

```python
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids, self.target_ids = [], []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=num_workers)
```

> `max_length` 与 `stride` 的关系：`stride < max_length` 时窗口**重叠**（增加样本量，但过拟合风险上升）；`stride = max_length` 时窗口不重叠。GPT-2 原版 `context_length = 1024`，本书教学用 256 以省显存。

### 3.7 创建词元嵌入

```python
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
```

- Embedding 层就是一个**可学习的权重矩阵**：行数 = 词汇表大小（50257），列数 = 词元向量维度（256）。
- 词元 ID 为 `i` 的词元向量 = 矩阵第 `i+1` 行（行号从 0 开始）。

### 3.8 编码单词位置信息

问题：同一个词元出现在序列不同位置，经 Embedding 后得到**完全相同**的向量，但语义可能不同（如 "the dog chased the cat" 中的两个 "the"）。

解决：**在词元嵌入上叠加位置嵌入**。

```python
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
input_embeddings = token_embeddings + pos_embeddings   # 相加
```

> 位置嵌入的两种实现：**绝对位置嵌入**（GPT-2 用，本书用）与 **相对位置嵌入**（如 RoPE，现代 LLM 常用）。

---

## 4. 第 3 章 编码注意力机制

> 对应代码：`ch03/01_main-chapter-code/ch03.ipynb`、`multihead-attention.ipynb`
> 本章完成路线图第一阶段的第 (2) 步。

### 4.1 长序列建模的问题

- 翻译任务中，源语言与目标语言语法结构不同，**无法逐词翻译**——某些词需要参考句中较早或较晚出现的词。
- **Transformer 之前**：RNN 是最流行的编码器-解码器架构。编码器逐步处理输入并更新隐状态，把整句含义压进**最后一个隐状态**；解码器用它逐字生成。
- **RNN 的核心缺陷**：解码阶段**无法直接访问编码器的早期隐状态**，只能依赖当前隐状态 → **长距离依赖会丢失**。
- **自注意力**解决之道：让序列中**每个位置与所有其他位置交互并权衡重要性**，计算出更高效的输入表示。

> 原文："Transformer 最早在《Attention is all you need》中被提出……模型引入了注意力机制，充分挖掘序列各节点之间的深度信息，并通过矩阵计算的并行化加速模型训练和推理性能。"

### 4.2 不带可训练权重的简化自注意力

给定输入序列 $x^{(1)}, \dots, x^{(T)}$（每个是 $d$ 维嵌入），目标是算出**上下文向量** $z^{(i)}$：

$$z^{(2)} = \sum_{i=1}^{T} \alpha_{2i} \cdot x^{(i)}$$

写成矩阵形式：$z = W \cdot x$，其中 $W$ 第 $i$ 行第 $j$ 列为 $\alpha_{ij}$。

- 上下文向量：**包含序列中所有元素信息**的嵌入，为每个词元创建丰富表示。
- 示例文本："Your journey starts with one step."

### 4.3 带可训练权重的自注意力

引入 3 个可训练参数矩阵 $W_q$、$W_k$、$W_v$：

$$q = W_q x,\quad k = W_k x,\quad v = W_v x$$

计算流程（以 $z^{(2)}$ 为例）：

1. **注意力得分**：$w_{2i} = q^{(2)} \cdot k^{(i)}$
2. **缩放 + Softmax** 得注意力权重：

$$\alpha_{2i} = \frac{e^{w_{2i}/\sqrt{d}}}{\sum_{j=1}^{T} e^{w_{2j}/\sqrt{d}}}$$

3. **加权求和值向量**：$z^{(2)} = \sum_{i=1}^{T} \alpha_{2i} \cdot v^{(i)}$

> **为什么除以 $\sqrt{d}$**：避免梯度过小，提升训练性能。这一技巧源自 Transformer 论文，也是"**缩放点积注意力**"中"缩放点积"的由来。

**代码清单 3-1（v1，用 `nn.Parameter`）：**

```python
import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        return attn_weights @ values
```

**v2：用 `nn.Linear` 替代参数矩阵**（无偏置时等价，但 `nn.Linear` 提供了**优化的权重初始化**，训练更稳定）：

```python
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys, queries, values = self.W_key(x), self.W_query(x), self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        return attn_weights @ values
```

**知乎笔记实机结果**（`d_in=3, d_out=2`，6 个词元）：

```
# SelfAttention_v1, seed=123
tensor([[0.2996, 0.8053], [0.3061, 0.8210], [0.3058, 0.8203],
        [0.2948, 0.7939], [0.2927, 0.7891], [0.2990, 0.8040]])
# SelfAttention_v2, seed=789
tensor([[-0.0739, 0.0713], [-0.0748, 0.0703], [-0.0749, 0.0702],
        [-0.0760, 0.0685], [-0.0763, 0.0679], [-0.0754, 0.0693]])
```

### 4.4 因果注意力（掩码注意力）

- **目的**：防止模型访问序列中**未来**的信息。语言建模中每个词的预测只能依赖之前出现的词。
- **实现**：把注意力得分矩阵**右上角置为负无穷大**（`-inf`），Softmax 后这些位置权重趋近 0；再重新归一化，使每行权重之和为 1。

**代码清单 3-3：带 dropout 的因果注意力**

```python
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys, queries, values = self.W_key(x), self.W_query(x), self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return attn_weights @ values
```

关于 **dropout**：

- 训练时**随机忽略一些隐藏层单元**，减少对特定单元的依赖，避免过拟合；
- **仅在训练期间使用**，推理时关闭（`model.eval()`）；
- Transformer 中通常在**两处**使用：① 计算注意力权重之后；② 权重作用于值向量之后。

**知乎笔记实机结果**（因果注意力权重，注意右上角全为 0）：

```
tensor([[[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.4833, 0.5167, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.3190, 0.3408, 0.3402, 0.0000, 0.0000, 0.0000],
         [0.2445, 0.2545, 0.2542, 0.2468, 0.0000, 0.0000],
         [0.1994, 0.2060, 0.2058, 0.1935, 0.1953, 0.0000],
         [0.1624, 0.1709, 0.1706, 0.1654, 0.1625, 0.1682]], ...])
```

### 4.5 多头注意力

- "多头"= 把注意力机制分成多个"头"，**每个头独立工作**，使用不同的线性投影。
- 两种实现：
  - **方案一（直观）**：堆叠多个 `CausalAttention` 模块，各自输出上下文向量后**拼接**；
  - **方案二（推荐）**：在**单个类**内通过**重塑张量形状**把维度拆成多个头，计算后再合并。

**代码清单 3-5：MultiHeadAttention**

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys, queries, values = self.W_key(x), self.W_query(x), self.W_value(x)

        # 拆头：(b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置 -> (b, num_heads, num_tokens, head_dim)
        keys, queries, values = (t.transpose(1, 2) for t in (keys, queries, values))

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)
```

**关键维度变化记忆表：**

| 张量 | 维度 |
|---|---|
| `x` | `(b, num_tokens, d_in)` |
| `keys/queries/values`（投影后） | `(b, num_tokens, d_out)` |
| 拆头后 | `(b, num_tokens, num_heads, head_dim)` |
| transpose 后 | `(b, num_heads, num_tokens, head_dim)` |
| `attn_scores` | `(b, num_heads, num_tokens, num_tokens)` |
| `context_vec`（合并后） | `(b, num_tokens, d_out)` |

> **为什么要加 `out_proj`**：把多个头的输出混合，让不同头之间的信息得以交互。

**GPT-2 配置**：`emb_dim = 768`、`n_heads = 12` → 每个头 `head_dim = 64`。

---

## 5. 第 4 章 从头实现 GPT 模型

> 对应代码：`ch04/01_main-chapter-code/ch04.ipynb`、`gpt.py`
> 本章完成路线图第一阶段的第 (3) 步。

### 5.1 GPT 配置字典

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # 词汇表大小
    "context_length": 256,   # 上下文长度（原版 1024，教学缩短）
    "emb_dim": 768,          # 嵌入维度
    "n_heads": 12,           # 注意力头数
    "n_layers": 12,          # Transformer 块数
    "drop_rate": 0.1,        # dropout 率
    "qkv_bias": False        # 查询-键-值偏置
}
```

### 5.2 用 `DummyGPTModel` 打通数据流

先用占位实现验证张量形状：

```python
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)    # torch.Size([2, 4, 50257])
```

### 5.3 层归一化（LayerNorm）

- 目的：**提高训练稳定性和效率**。把激活调整为**均值为 0、方差为 1**。
- 与批归一化的区别：LayerNorm **对单个样本的嵌入维度做归一化**，不跨批次。

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

- `eps` 防止除零；
- `scale` / `shift` 是**可训练参数**，让模型自行学习最适合的缩放与偏移。

### 5.4 GELU 激活函数

- GELU、SwiGLU 比 ReLU 更平滑，通常能提升性能。
- 精确形式：$\text{GELU}(x) = x \cdot \Phi(x)$，$\Phi$ 是标准高斯分布的 CDF。
- 实际用**计算量更小的近似**（GPT-2 原版即用此）：

$$\text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}} \cdot \left(x + 0.044715 \cdot x^3\right)\right]\right)$$

**GELU vs ReLU：**

| | ReLU | GELU |
|---|---|---|
| 形状 | 分段线性，0 点有**尖锐拐角** | 平滑非线性 |
| 负值 | 输出恒为 0 | 输出**小的非零值** |
| 梯度 | 负区间梯度为 0 | 几乎所有负值都有非零梯度 |
| 优化 | 有时更困难 | 参数可做更细微的调整 |

### 5.5 前馈网络（FeedForward）

结构：`Linear(768→3072) → GELU → Linear(3072→768)`

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
```

> 关键设计：输入输出维度**保持一致**，但中间**扩展到 4 倍**（768 → 3072），让模型探索更丰富的表示空间。

### 5.6 残差连接（快捷连接）

- 把某层的**输入直接叠加到该层输出**上（`x = x + shortcut`）。
- 解决**深层网络梯度消失**问题，是训练深层 Transformer 的关键。

### 5.7 Transformer 块

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            context_length=cfg["context_length"], num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 注意力块 + 残差
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # 前馈块 + 残差
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
```

> **核心思想**：自注意力（多头）负责**识别序列中元素之间的关系**；前馈网络在**每个位置上独立地修改数据**。二者结合提升模型处理复杂模式的能力。
> **注意**：这里是 **Pre-LN**（先归一化再进子层），比原始 Transformer 的 Post-LN 更易训练。

### 5.8 完整的 GPT 模型

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)
```

**参数量之谜（重要！）：**

```python
print("Total number of parameters:", total_params)   # 163,009,536
```

为什么是 1.63 亿而不是 1.24 亿？因为原始 GPT-2 用了 **权重共享（weight tying）**——把词元嵌入层**复用**为输出层。

```python
total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
# Number of trainable parameters considering weight tying: 124,412,160
```

> 原书作者观点：**使用独立的词元嵌入层和输出层能获得更好的训练效果**，因此 `GPTModel` 里用了独立的层；但第 6 章加载 OpenAI 预训练权重时会再回到权重共享的概念。
> 1.63 亿参数按 32 位浮点（4 字节/参数）计算，约需 **621 MB** 内存。

### 5.9 文本生成

`generate_text_simple`（贪婪解码）：

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]          # 超出上下文则截断
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]                  # 只看最后一个时间步
        probas = torch.softmax(logits, dim=-1)     # 转概率
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)    # 拼回输入
    return idx
```

**生成流程（图 25/26）**：从 "Hello, I am" 开始，每轮预测下一个词元并追加到上下文，逐步生成 "Hello, I am a model ready to help."。

---

## 6. 第 5 章 在无标签数据上预训练

> 对应代码：`ch05/01_main-chapter-code/ch05.ipynb`、`gpt_train.py`、`gpt_generate.py`
> 本章完成路线图第二阶段的第 (4)(5) 步。

### 6.1 用交叉熵损失评估生成质量

未训练的模型输出是**胡言乱语**：

```python
start_context = "Every effort moves you"
# Output text:
# Every effort moves you rentingetic wasn refres RexMeCHicular stren
```

**交叉熵损失**衡量真实分布与预测分布的差异：

$$L_{CE} = -\sum_{i=1}^{n} y_i \log \hat{y}_i$$

- $n$：词汇表中词元的数量；$y_i$ 是真实概率（one-hot，目标词元为 1，其余为 0）；
- $\hat{y}_i$ 是模型预测概率；$\hat{y}_i$ 越接近 1，$L_{CE}$ 越小。

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten())
    return loss
```

> **困惑度（perplexity）= exp(loss)**，是更直观的评估指标：困惑度 48725 表示模型在约 48725 个词元中"犹豫不决"。

### 6.2 训练大语言模型

```python
def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen
```

**配置与运行**：

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257, "context_length": 256, "emb_dim": 768,
    "n_heads": 12, "n_layers": 12, "drop_rate": 0.1, "qkv_bias": False
}
OTHER_SETTINGS = {
    "learning_rate": 0.0004, "num_epochs": 10,
    "batch_size": 2, "weight_decay": 0.1
}
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=0.0004, weight_decay=0.1)
# 数据集按 90% / 10% 切分为训练/验证
```

**知乎笔记实机训练输出（关键证据：模型从乱码到语法正确）：**

```
Ep 1 (Step 000000): Train loss 9.781, Val loss 9.933
Ep 1 (Step 000005): Train loss 8.111, Val loss 8.339
Every effort moves you,,,,,,,,,,,,.

Ep 2 (Step 000015): Train loss 5.961, Val loss 6.616
Every effort moves you, and, and, and, and, and, and, ...

Ep 3 (Step 000025): Train loss 5.201, Val loss 6.348
Every effort moves you, and I had been.

Ep 4 (Step 000035): Train loss 4.069, Val loss 6.226
Every effort moves you know the "I he had the donkey and I had the ...

Ep 8 (Step 000070): Train loss 0.985, Val loss 6.242
Ep 9 (Step 000080): Train loss 0.541, Val loss 6.393
Ep 10 (Step 000085): Train loss 0.391, Val loss 6.452
Every effort moves you know," was one of the axioms he laid down across
the Sevres and silver of an exquisitely appointed luncheon-table, ...
```

**过拟合分析（原书原文）**："在训练开始阶段，训练集损失和验证集损失急剧下降……然而，在第二轮之后，训练集损失继续下降，验证集损失则停滞不前。这表明模型仍在学习，但在第二轮之后开始对训练集过拟合。这种记忆现象其实是可以预料到的，因为我们使用了一个非常小的训练数据集……通常，在更大的数据集上训练模型时，只训练一轮是很常见的做法。"

### 6.3 控制随机性的解码策略

问题：`argmax`（贪婪解码）每次输出**完全相同**，但实际使用 LLM 时同一输入会得到不同输出。

**① 概率采样**——用 `torch.multinomial` 替换 `argmax`：

```python
next_token_id = torch.multinomial(probas, num_samples=1).item()
```

**② 温度缩放（temperature scaling）**——把 logits 除以 temperature 再做 Softmax：

```python
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)
```

| temperature | 分布 | 效果 |
|---|---|---|
| **< 1**（如 0.1） | 更尖锐 | 几乎总是选最可能的词元，接近 `argmax` |
| **= 1** | 原始分布 | — |
| **> 1**（如 5） | 更均匀 | 更多变化，但更容易生成无意义文本 |

**③ Top-k 采样**——只保留概率最高的 k 个词元再采样。

**完整实现：**

```python
def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:                       # Top-k 过滤
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val,
                                 torch.tensor(float("-inf")).to(logits.device),
                                 logits)

        if temperature > 0.0:                       # 温度缩放 + 采样
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:                                       # 贪婪解码
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:                      # 遇到 EOS 提前停止
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

**知乎笔记实机结果**（`temperature=1.4, top_k=25`）：

```
Every effort moves you stand to work on surprise, a one of us had gone with random-
```

### 6.4 保存与加载模型权重

```python
torch.save(model.state_dict(), "model.pth")

model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", weights_only=True))
```

> **重要**：PyTorch 保存的是 `state_dict`（参数张量），**不含模型架构**。加载时必须先用相同配置实例化 `GPTModel`。

### 6.5 从 OpenAI 加载预训练权重

这是本书最实用的技巧——**跳过昂贵的预训练**。

```python
from gpt_download import download_and_load_gpt2
from chapter04 import GPTModel

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
print(settings)
# {'n_vocab': 50257, 'n_ctx': 1024, 'n_embd': 768, 'n_head': 12,
#  'n_layer': 12, ...}

model = GPTModel(GPT_CONFIG_124M)
load_weights_into_gpt(model, params)
```

**`load_weights_into_gpt` 的核心工作：**

1. 把 OpenAI 的权重名（`wte`、`wpe`、`attn/c_attn`、`mlp/c_fc` 等）映射到本书 `GPTModel` 的层名；
2. 拆分 `c_attn` 的合并 QKV 权重；
3. **处理权重共享**——把 `wte` 同时赋给 `tok_emb` 和 `out_head`。

```python
def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1)
        # ... 逐个 assign 到 W_query / W_key / W_value
        # ... 前馈层 c_fc / c_proj
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])  # 权重共享
```

**验证加载成功**：

```python
# Every effort moves you forward.
# The first step is to understand the importance of your work
```

> 加载后可达到 **124M** 参数（含权重共享）。

---

## 7. 第 6 章 针对分类的微调

> 对应代码：`ch06/01_main-chapter-code/ch06.ipynb`、`gpt_class_finetune.py`
> 本章完成路线图第三阶段的第 (8) 步。

### 7.1 分类微调 vs 指令微调

| | 分类微调 | 指令微调 |
|---|---|---|
| 输出层 | 替换为**类别数**个输出节点 | 保持词表大小输出 |
| 输入 | 文本（无需额外指令） | 指令 + 输入 |
| 能力 | **只能预测训练时见过的类别** | 能执行**更广泛**的任务 |
| 数据/算力 | 需求较少 | 需求更大 |

> 原文："经过分类微调的模型只能预测它在训练过程中遇到的类别……它不能对输入文本进行其他分析或说明。"

### 7.2 准备数据集（SMS Spam Collection）

```python
import urllib.request, zipfile, os
from pathlib import Path
import pandas as pd

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
# 下载解压后读取
df = pd.read_csv("sms_spam_collection/SMSSpamCollection.tsv",
                 sep="\t", header=None, names=["Label", "Text"])

print(df["Label"].value_counts())
# ham     4825
# spam     747
```

**类别极不平衡**（ham 是 spam 的 6.5 倍）→ **下采样**到每类 747 条：

```python
def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    return pd.concat([ham_subset, df[df["Label"] == "spam"]])

balanced_df = create_balanced_dataset(df)
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
```

**划分 70% / 10% / 20%**（训练/验证/测试）：

```python
def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    return df[:train_end], df[train_end:validation_end], df[validation_end:]
```

### 7.3 创建数据加载器

**填充策略选择**（原文对比）：

- 方案一：截断到最短长度 → 计算开销小，但**可能丢信息**；
- 方案二：**填充到最长长度** → 保留全部内容。**本书选方案二**。

```python
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [t[:self.max_length] for t in self.encoded_texts]

        self.encoded_texts = [
            t + [pad_token_id] * (self.max_length - len(t))
            for t in self.encoded_texts]

    def __getitem__(self, index):
        return (torch.tensor(self.encoded_texts[index], dtype=torch.long),
                torch.tensor(self.data.iloc[index]["Label"], dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        return max(len(t) for t in self.encoded_texts)
```

> 填充词元用 `<|endoftext|>` 的 ID **50256**（性能考虑：直接拼 ID，不拼字符串）。

**实机结果：**

```
train_dataset.max_length = 120          # 最长序列不超过 120 个词元
Input batch dimensions: torch.Size([8, 120])
Label batch dimensions torch.Size([8])
130 training batches / 19 validation batches / 38 test batches
```

### 7.4 加载预训练模型 + 添加分类头

```python
CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG = {"vocab_size": 50257, "context_length": 1024,
               "drop_rate": 0.0, "qkv_bias": True}
model_configs = {
    "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
```

**替换输出层（768 → 2）**：

```python
for param in model.parameters():
    param.requires_grad = False        # 先全部冻结

torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(in_features=768, out_features=num_classes)

# 额外解冻最后一个 Transformer 块 + 最终 LayerNorm（显著提升性能）
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True
```

> **为什么只微调最后几层**：神经网络**较低层捕捉基础语言结构和语义**（通用），**最后几层更侧重细微语言模式和特定任务特征**。只微调最后几层通常足以适应新任务，且计算更高效。
> **为什么输出节点数 = 类别数**（而非二分类用 1 个）：更通用，可无缝扩展到三分类、多分类。

### 7.5 为什么只取最后一个词元的输出？

```python
with torch.no_grad():
    outputs = model(inputs)
print(outputs.shape)              # torch.Size([1, 4, 2])

print("Last output token:", outputs[:, -1, :])
# Last output token: tensor([[-3.5983,  3.9902]])
```

**关键原因**：因果注意力掩码下，**序列中最后一个词元累积了最多的信息**——它是唯一能访问前面所有词元的词元。

### 7.6 计算分类损失和准确率

```python
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    # ... 遍历批次
    #     logits = model(input_batch)[:, -1, :]
    #     predicted_labels = torch.argmax(logits, dim=-1)
    #     correct_predictions += (predicted_labels == target_batch).sum().item()
    return correct_predictions / num_examples
```

**分类损失**（相比预训练，**只关注最后一个词元**）：

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]        # ← 唯一改动
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss
```

**微调前的初始表现（接近随机猜）：**

```
Training accuracy: 46.25%
Validation accuracy: 45.00%
Test accuracy: 48.75%

Training loss: 2.453
Validation loss: 2.583
Test loss: 2.322
```

### 7.7 微调结果

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 5
train_losses, val_losses, train_accs, val_accs, examples_seen = \
    train_classifier_simple(model, train_loader, val_loader, optimizer, device,
                            num_epochs=num_epochs, eval_freq=50, eval_iter=5)
```

**实机输出（MacBook Air M3 约 6 分钟；V100/A100 不到半分钟）：**

```
Ep 1 (Step 000000): Train loss 2.153, Val loss 2.392
Ep 1 (Step 000100): Train loss 0.523, Val loss 0.557
Training accuracy: 70.00% | Validation accuracy: 72.50%
Ep 2 (Step 000250): Train loss 0.409, Val loss 0.353
Training accuracy: 82.50% | Validation accuracy: 85.00%
Ep 3 (Step 000350): Train loss 0.340, Val loss 0.306
Training accuracy: 90.00% | Validation accuracy: 90.00%
Ep 4 (Step 000500): Train loss 0.222, Val loss 0.137
Training accuracy: 100.00% | Validation accuracy: 97.50%
Ep 5 (Step 000600): Train loss 0.083, Val loss 0.074
Training accuracy: 100.00% | Validation accuracy: 97.50%
Training completed in 5.65 minutes.

# 全量评估
Training accuracy: 97.21%
Validation accuracy: 97.32%
Test accuracy: 95.67%
```

> **轮数选择**：原文说"5 轮是一个不错的起点。如果模型在前几轮之后出现过拟合，则可能需要减少轮数；如果验证集损失仍可能改善，则应增加轮数。"

### 7.8 使用分类器

```python
def classify_review(text, model, tokenizer, device, max_length=None,
                    pad_token_id=50256):
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]
    input_ids = input_ids[:min(max_length, supported_context_length)]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()
    return "spam" if predicted_label == 1 else "not spam"
```

验证：

```python
text_1 = ("You are a winner you have been specially"
          " selected to receive $1000 cash or a $2000 award.")
print(classify_review(text_1, model, tokenizer, device,
                      max_length=train_dataset.max_length))   # spam

text_2 = ("Hey, just wanted to check if we're still on"
          " for dinner tonight? Let me know!")
print(classify_review(text_2, model, tokenizer, device,
                      max_length=train_dataset.max_length))   # not spam

torch.save(model.state_dict(), "review_classifier.pth")
```

---

## 8. 第 7 章 通过微调遵循人类指令

> 对应代码：`ch07/01_main-chapter-code/ch07.ipynb`、`gpt_instruction_finetuning.py`、`ollama_evaluate.py`
> 本章完成路线图第三阶段的第 (9) 步。

### 8.1 准备指令数据集

数据集 `instruction-data.json`，**1100 条**指令-回复对（仅 204 KB）：

```python
data = download_and_load_file("instruction-data.json", url)
print("Number of entries:", len(data))          # 1100

print(data[50])
# {'instruction': 'Identify the correct spelling of the following word.',
#  'input': 'Ocassion', 'output': "The correct spelling is 'Occasion.'"}

print(data[999])
# {'instruction': "What is an antonym of 'complicated'?",
#  'input': '', 'output': "An antonym of 'complicated' is 'simple'."}
```

**两种提示词风格：**

| 风格 | 特点 |
|---|---|
| **Alpaca**（本书默认） | 用 `### Instruction:` / `### Input:` / `### Response:` 结构化分节 |
| **Phi-3** | 更简单，使用特殊词元 `<|user|>` / `<|assistant|>` |

```python
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text
```

**划分 85% / 5% / 10%：**

```python
train_portion = int(len(data) * 0.85)     # 935
test_portion = int(len(data) * 0.1)       # 110
val_portion = len(data) - train_portion - test_portion   # 55
```

### 8.2 自定义批处理（本章核心难点）

**5 个子步骤（图 7-6）：**
1. 应用提示词模板
2. 分词
3. 添加填充词元
4. 创建目标词元 ID
5. 用 **-100** 掩码填充词元

**① InstructionDataset（预分词）：**

```python
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            full_text = format_input(entry) + f"\n\n### Response:\n{entry['output']}"
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)
```

**② 自定义聚合函数（custom_collate_fn）：**

```python
def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100,
                      allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs = torch.tensor(padded[:-1])     # 输入：截掉最后一个
        targets = torch.tensor(padded[1:])     # 目标：左移一位

        # 除第一个填充词元外，其余填充词元替换为 -100
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    return torch.stack(inputs_lst).to(device), torch.stack(targets_lst).to(device)
```

**输出验证：**

```
inputs = [[0, 1, 2, 3, 4], [5, 6, 50256, 50256, 50256], [7, 8, 9, 50256, 50256]]
targets= [[1, 2, 3, 4, 50256], [6, 50256, -100, -100, -100], [8, 9, 50256, -100, -100]]
```

**③ 为什么是 -100？（关键原理）**

PyTorch 交叉熵损失的默认参数是 `cross_entropy(..., ignore_index=-100)`——**它会忽略标记为 -100 的目标**。

```python
logits_1  = torch.tensor([[-1.0, 1.0], [-0.5, 1.5]])
targets_1 = torch.tensor([0, 1])
print(torch.nn.functional.cross_entropy(logits_1, targets_1))   # tensor(1.1269)

logits_2  = torch.tensor([[-1.0, 1.0], [-0.5, 1.5], [-0.5, 1.5]])
targets_3 = torch.tensor([0, 1, -100])
print(torch.nn.functional.cross_entropy(logits_2, targets_3))   # tensor(1.1269) ← 相同！
```

> **保留第一个 50256 的原因**：帮助模型学会**何时生成结束符**，以在适当时候结束回复。
> **注意**：分类微调**不需要**这步，因为它只根据最后一个输出词元训练。

**关于是否掩码指令部分**：研究者仍有分歧。Shi 等人 2024 年论文《Instruction Tuning With Loss Over Instructions》指出**不掩码指令可以提升性能**；本书**不掩码**（留作练习 7.2）。

### 8.3 数据加载器

```python
from functools import partial
customized_collate_fn = partial(custom_collate_fn, device=device,
                                allowed_max_length=1024)

train_loader = DataLoader(InstructionDataset(train_data, tokenizer),
                          batch_size=8, collate_fn=customized_collate_fn,
                          shuffle=True, drop_last=True, num_workers=0)
# val_loader / test_loader 同理（shuffle=False, drop_last=False）
```

**实机输出（批次长度可变——这正是自定义聚合函数的威力）：**

```
Train loader:
torch.Size([8, 61]) torch.Size([8, 61])
torch.Size([8, 76]) torch.Size([8, 76])
torch.Size([8, 73]) torch.Size([8, 73])
...
```

> **设备设置技巧**：把 `.to(device)` 写在聚合函数里，可在训练循环之外的后台执行，**避免阻塞 GPU**。
> Apple Silicon 可用 `device = torch.device("mps")`（但 PyTorch 支持仍属实验阶段，数值结果可能有差异）。

### 8.4 加载预训练模型

**本章改用 `gpt2-medium (355M)`**：

> 原文："这是因为参数量为 1.24 亿的模型容量过于有限，无法通过指令微调获得令人满意的结果。具体来说，较小的模型在学习高质量的指令遵循任务时，缺乏执行该任务所需的复杂模式和细微行为的能力。"

```python
CHOOSE_MODEL = "gpt2-medium (355M)"
settings, params = download_and_load_gpt2(model_size="355M", models_dir="gpt2")
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
```

> ⚠️ 下载 gpt2-medium 需要约 **1.42 GB** 存储，是最小 GPT 模型的 3 倍。

### 8.5 训练与评估

训练流程与第 6 章类似，但使用 `train_model_simple` 风格的函数（因为目标是生成而非分类），损失函数回到**全词元交叉熵**。

**评估方式**：把测试集的模型输出保存，用 **Ollama** 调用另一个 LLM（如 Llama 3）作为裁判打分（`ollama_evaluate.py`）。

```python
def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    ...
```

---

## 9. 附录 A~F

### 附录 A　PyTorch 简介

- 三大核心组件、张量基础（标量/向量/矩阵/张量）、张量数据类型与常用操作；
- 把模型视为**计算图**、自动微分、多层神经网络实现；
- 高效数据加载器（`Dataset` / `DataLoader`）、典型训练循环；
- 保存与加载模型；**GPU 优化训练**（单 GPU / 多 GPU DDP，`DDP-script.py`）。
- 代码：`appendix-A/01_main-chapter-code/code-part1.ipynb`、`code-part2.ipynb`

### 附录 B　参考文献和延伸阅读

按章节列出参考文献（第 1~7 章 + 附录 A），是深入学习的索引。

### 附录 C　练习的解决方案

覆盖第 2~7 章与附录 A 的习题解答。

### 附录 D　为训练循环添加更多细节和优化功能

在 `train_model_simple` 基础上引入 **3 项稳定训练的技术**：

**D.1 学习率预热（warmup）**

- 作用：把学习率从很小的初始值**逐步提升**到峰值，降低训练初期大幅度不稳定更新的风险。
- 预热步数通常设为总步数的 **0.1% ~ 20%**。

```python
total_steps = len(train_loader) * n_epochs
warmup_steps = int(0.2 * total_steps)      # 20% 预热 → 27 步
lr_increment = (peak_lr - initial_lr) / warmup_steps

if global_step < warmup_steps:
    lr = initial_lr + global_step * lr_increment
else:
    lr = peak_lr
```

**D.2 余弦衰减（cosine decay）**

- 预热后按**半个余弦周期**逐渐降低学习率，减缓权重更新速度，避免越过损失最小值。

```python
progress = (global_step - warmup_steps) / (total_training_steps - warmup_steps)
lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
```

**D.3 梯度裁剪（gradient clipping）**

- 设定阈值，超过的梯度被缩放到预定最大值，确保参数更新在可控范围内。
- L2 范数：$\|v\|_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$；缩放因子 = `max_norm / ||G||₂`。

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# 裁剪前最高梯度: tensor(0.0411)  →  裁剪后: tensor(0.0185)
```

**D.4 修改后的训练函数** `train_model`：三者合一（预热 + 余弦衰减 + 梯度裁剪）。

### 附录 E　使用 LoRA 进行参数高效微调

**核心思想**：不直接学习 $\Delta W$，而是用两个小矩阵近似：

$$W_{updated} = W + \Delta W \approx W + A \cdot B$$

其中 $A \in \mathbb{R}^{d \times r}$、$B \in \mathbb{R}^{r \times k}$，**$r \ll d, k$**（$r$ 是可调超参数）。

**关键优势**：$W$ 与 $AB$ **保持分离**，预训练权重可保持不变，只需保存很小的 LoRA 矩阵 → 定制更灵活、存储更省。

```python
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))   # B 初始化为 0
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            replace_linear_with_lora(module, rank, alpha)
```

> **为什么 B 初始化为 0**：保证训练开始时 $AB = 0$，即**不改变原始权重**，训练无缝衔接。

**参数量对比（rank=16, alpha=16）：**

```
Total trainable parameters before: 124,441,346
Total trainable parameters after:  0            # 先全部冻结
Total trainable LoRA parameters:   2,666,528    # 约原来的 1/50
```

**LoRA 微调结果：**

```
Ep 1 (Step 000100): Train loss 0.111, Val loss 0.229
Training accuracy: 97.50% | Validation accuracy: 95.00%
Ep 5 (Step 000600): Train loss 0.000, Val loss 0.056
Training accuracy: 100.00% | Validation accuracy: 97.50%
Training completed in 12.10 minutes.

# 全量评估
Training accuracy: 100.00%
Validation accuracy: 96.64%
Test accuracy: 98.00%
```

> 原文提示：本示例中 LoRA 反而**更慢**（前向传播引入额外计算），但**对大模型**，反向传播成本更高，LoRA 通常更快。`rank` 和 `alpha` 设为 16 是不错的默认值；通常把 `alpha` 设为 `rank` 的**一半、两倍或相等**。

### 附录 F　理解推理大语言模型

> 作者博客文章，作为中文版附赠内容收录。

**"推理"的定义**：解答需要**复杂、多步骤生成并包含中间过程**的问题。回答"法国首都是哪里"不涉及推理；"火车以 60 英里/小时行驶 3 小时能走多远"需要推理。

**推理模型的中间步骤呈现方式**：① 直接展示在回答中（用户可见完整推理）；② 内部多轮迭代但不展示（如 OpenAI o1）。

**何时使用推理模型**：多步推理的复杂任务（解谜题、高级数学推导、复杂编程）。对总结/翻译/知识问答等简单任务**不必用**——会带来不必要的开销，还可能因"过度思考"而更易出错。

**DeepSeek R1 的三个变体：**

| 模型 | 训练流程 |
|---|---|
| **DeepSeek-R1-Zero** | 基于 DeepSeek-V3（6710 亿），**纯强化学习**（无 SFT），"冷启动" |
| **DeepSeek-R1** | 在 R1-Zero 基础上增加**监督微调 + 强化学习** |
| **DeepSeek-R1-Distill** | 用 R1 生成的数据微调 Qwen / Llama 小模型 |

**构建推理模型的四大核心方法：**

**① 推理时间扩展（inference-time scaling）**
- 不修改/不重新训练模型，而是**增加推理时的计算资源**。
- 方法：思维链提示（"think step by step"）、多数投票、束搜索等搜索算法。
- 缺点：推理成本上升，大规模部署更贵。

**② 纯强化学习**
- DeepSeek-R1-Zero **完全通过 RL 训练**，跳过 SFT。
- 两种奖励：**准确性奖励**（LeetCode 编译器验证代码、确定性系统评估数学）、**格式奖励**（确保 `<think>` 标签等格式）。
- 出现了 **"Aha" 时刻**——模型自发学会生成推理过程。

**③ 监督微调 + 强化学习（SFT + RL）**
- DeepSeek-R1 的路线，也是构建高性能推理模型的**首选方法**。
- 流程：用 R1-Zero 生成"冷启动"SFT 数据 → 指令微调 → RL（准确性 + 格式 + **一致性奖励**，避免混用多语言）→ 再收集 60 万思维链 + 20 万知识 SFT 样本 → 指令微调 V3 → 最后一轮 RL。

**④ 纯监督微调与蒸馏**
- 用 R1 的 SFT 数据集微调小模型（Llama 8B/70B、Qwen 1.5B~32B）。
- 优点：小模型**效率高、成本低**，可在低端硬件运行。
- 局限：**不能推动创新**，始终依赖更强的现有模型。

**一个重要实验结论**：在较小模型（Qwen-32B）上，**蒸馏远比纯强化学习有效**——纯 RL 可能不足以在小模型中激发强大推理能力。

**低成本实践案例：**

| 项目 | 方法 | 成本 |
|---|---|---|
| **Sky-T1** | 纯 SFT，1.7 万样本，32B 模型 | **450 美元** |
| **TinyZero** | 纯 RL，复刻 R1-Zero，3B 模型 | **< 30 美元** |
| **旅程学习（Journey Learning）** | SFT 中**加入错误解题路径**，让模型从错误中学习（vs 传统"捷径学习"只学正确路径） | — |

---

## 10. 超参数速查表

### 模型配置

| 配置项 | GPT-2 small (124M) | GPT-2 medium (355M) | GPT-2 large (774M) | GPT-2 XL (1558M) |
|---|---|---|---|---|
| `vocab_size` | 50257 | 50257 | 50257 | 50257 |
| `context_length` | 1024 | 1024 | 1024 | 1024 |
| `emb_dim` | 768 | 1024 | 1280 | 1600 |
| `n_heads` | 12 | 16 | 20 | 25 |
| `n_layers` | 12 | 24 | 36 | 48 |
| `drop_rate` | 0.0 | 0.0 | 0.0 | 0.0 |
| `qkv_bias` | True | True | True | True |

> 教学版预训练把 `context_length` 缩短到 **256** 以节省显存；加载 OpenAI 权重时恢复为 **1024**。

### 训练超参数

| 场景 | 学习率 | 轮数 | 批次大小 | weight_decay | 优化器 |
|---|---|---|---|---|---|
| 预训练（第 5 章） | 4e-4 | 10 | 2 | 0.1 | AdamW |
| 分类微调（第 6 章） | 5e-5 | 5 | 8 | 0.1 | AdamW |
| LoRA 微调（附录 E） | 5e-5 | 5 | 8 | 0.1 | AdamW |
| 附录 D 改进版 | peak 5e-4，initial 1e-5，min 1e-5 | 15 | 2 | 0.1 | AdamW + warmup + cosine + clip |

### 关键数值

| 项目 | 数值 |
|---|---|
| GPT-2 BPE 词汇表大小 | 50257 |
| `<\|endoftext\|>` 词元 ID | **50256** |
| 交叉熵 `ignore_index` | **-100** |
| The Verdict 字符数 / 词元数 | 20479 / 5145（BPE） |
| The Verdict 简易分词词元数 / 词汇表 | 4690 / 1130（+2 特殊词元 = 1132） |
| GPT-2 small 参数量（含权重共享） | 124,412,160 |
| GPTModel 实际参数量（独立输出层） | 163,009,536 |
| 分类数据集最长序列 | 120 词元 |
| 指令数据集条数 / 划分 | 1100 / 935 + 55 + 110 |
| LoRA 可训练参数（rank=16） | 2,666,528 |

---

## 11. 踩坑清单

| 现象 | 原因 | 解决 |
|---|---|---|
| `KeyError: 'Hello'` | 用 `SimpleTokenizerV1` 编码训练集外的词 | 换 `SimpleTokenizerV2`（`<\|unk\|>`）或直接用 tiktoken BPE |
| `RuntimeError: num_tokens > context_length` | 输入序列超过位置嵌入的最大长度 | 截断到 `context_length`，或用 `allowed_max_length` |
| 参数量是 1.63 亿而非 1.24 亿 | 没有做权重共享 | 减去 `out_head` 参数量；加载 OpenAI 权重时 `out_head.weight = wte` |
| 训练后验证损失不降 | 数据集太小（The Verdict 仅 5145 词元）+ 多轮训练 | 这是**预期的过拟合**；换大数据集，通常只训 1 轮 |
| 模型推理结果随机 | 忘了 `model.eval()` | 推理前调用，关闭 dropout |
| 微调准确率约 50% | 只微调了 `out_head` 或加载权重失败 | 额外解冻最后一个 Transformer 块 + `final_norm` |
| 分类准确率算不准 | 用了全部词元的输出 | 分类任务只取 `model(x)[:, -1, :]` |
| 指令微调损失异常 | 填充词元参与了损失计算 | 把填充目标替换为 **-100** |
| 显存/内存不够 | `context_length`、`batch_size` 太大 | 缩小到 256/2，或用 LoRA |
| 加载 `.pth` 报错 | `state_dict` 不含架构 | 先用相同 `GPT_CONFIG` 实例化 `GPTModel` |
| 多轮生成结果每次相同 | 用了 `argmax`（贪婪解码） | 改用 `multinomial` + 温度缩放 + top-k |

---

## 12. 延伸阅读

**原书引用与官方资源**

- 书籍主页：https://www.manning.com/books/build-a-large-language-model-from-scratch
- 官方代码：https://github.com/rasbt/LLMs-from-scratch
- 配套视频课程：17 小时 15 分钟，按章节组织
- 免费测验 PDF：《Test Yourself On Build a Large Language Model (From Scratch)》，170 页，每章约 30 题
- 作者博客：https://sebastianraschka.com/

**关键论文**

| 论文 | 主题 |
|---|---|
| *Attention Is All You Need* (2017) | Transformer 原始论文 |
| *Improving Language Understanding by Generative Pre-Training* | GPT 原始论文 |
| *Language Models are Few-Shot Learners* | GPT-3 |
| *Training language models to follow instructions with human feedback* | InstructGPT / RLHF |
| *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al.) | LoRA |
| *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL* | 推理模型 |
| *Scaling LLM Test-Time Compute Optimally...* | 推理时间扩展 |
| *Instruction Tuning With Loss Over Instructions* (Shi et al., 2024) | 指令掩码 |
| *O1 Replication Journey: A Strategic Progress Report – Part 1* | 旅程学习 |

**数据集**

- The Verdict（预训练文本）：`ch02/01_main-chapter-code/the-verdict.txt`
- SMS Spam Collection（分类）：https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
- 指令数据集：`ch07/01_main-chapter-code/instruction-data.json`

---

*本笔记基于原书 PDF、官方代码仓库 rasbt/LLMs-from-scratch 与知乎读书笔记交叉整理，代码与数值均标注了来源。建议配合原书章节与仓库 Notebook 对照阅读、动手运行。*
