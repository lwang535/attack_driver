
# Explore and visualization
### 正负样本比的统计

->>>情况：正负样本严重失衡，比例在7:93的样子

->>>>解决方法：下采样（未完成）
### 问题长度分布的统计

->>>情况：负样本是明显的长尾分布，99%的问题词数都少于30；正样本有点像均匀分布，99%的问题词数在5-45之间。

->>>思考：model1对每个问题仅取前30词是否会削弱对正样本的预测？

->>>解决方法：将词数放大到46或者50词（未完成）
### 其他的feature engineering，比如tf-idf（未完成）


# Model
### 第一个submission：0.564

#### 外部参数：

样本数：10万；embedding 使用：glove.840B.300d.txt；自变量矩阵：30\*300*问题数

#### 内部参数：

batch_size = 90; steps_per_epoch = 100; epochs = 10; validation set的比例0.3

#### 模型结构：

Conv1D+LSTM+Dense

#### Evaluation
Accuracy 有0.95，AUC 也挺高，估计有过拟合问题。

#### Conclusion
1、换用别的embedding? 

2、
