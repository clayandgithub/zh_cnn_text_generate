# 基于cnn的中文文本生成算法

## 简介
参考[IMPLEMENTING A CNN FOR TEXT CLASSIFICATION IN TENSORFLOW]((http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/))实现的一个简单的卷积神经网络，用于中文文本生成（此项目使用的测试文本是古龙的小说《长生剑》）。

## 运行方法

### 训练
run `python train.py` to train the cnn with a text file (only support chinese!) (change the config filepath in FLAGS to your own)

### 在tensorboard上查看summaries
run `tensorboard --logdir /{PATH_TO_CODE}/runs/{TIME_DIR}/summaries/` to view summaries in web view

### 测试生成
run `python eval.py`

### 说明
在目前的网络参数下，结果并不好，生成的句子都不通顺。等将来我对RNN和CNN的理解更进一步之后再来完善此项目吧。
