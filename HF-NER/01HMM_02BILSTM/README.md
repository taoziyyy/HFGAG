# NER-English
A small English named entity recognition task
## 模型
使用了HMM以及Bilstm两种方法。
### HMM
HMM 的训练部分采用了监督式的极大似然估计。
### BiLstM
采用了pytorch框架中的LSTM模型。
## 数据
原始数据为json形式，包含实体和位置，为非BIO格式，在data change.py下变更为BIO模式，分别通过HMM，Bilstm模型进行训练后，将结果以两种形式输出到My output results中。
