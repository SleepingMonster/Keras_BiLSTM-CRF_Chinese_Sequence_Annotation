# Keras_BiLSTM-CRF_Chinese_Sequence_Annotation
中山大学自然语言处理项目：中文分词（序列标注/命名实体识别）。Keras实现，BiLSTM+CRF框架。
## Readme

#### 实验环境

- `keras` 2.3.1版本和`tensorflow` 2.2版本（或者其他相匹配的`keras`和`tensorflow`版本）

- `keras_contrib`库、`gensim`库、`pickle`库、`tqdm`库

#### 实验工具

jupyter notebook

#### 文件组织

- 词向量：`sgns.context.word-character.char1-1` 文件
  - 来自于 https://github.com/Embedding/Chinese-Word-Vectors 中“Co-occurrence Type”中的“Word->Character(1)”的"Context word vectors"
- `main`文件：
  - `main.ipynb`（jupyter notebook格式）和`main.py`，建议执行`main.ipynb`。
  - 逐块执行即可得到训练`EPOCH`次数之后的测试集的预测结果，并会输出F1值。
    - `EPOCH`的取值可在第二块的超参量部分进行调整。
  - 结果：训练集迭代运行5次之后的结果保存为`msr_test_predict.txt`和`msr_test_predict.utf8`两种格式的文件，可对比其与`msr_test_gold`的结果，得到对应的F1值为0.9302。
  - **注意：**由于一开始模型的其他参数的初始化的随机性，不同时间训练相同的EPOCH次数可能得到不同的F1值，但整体上不会有较大差异，大概在1个百分点之内。）
