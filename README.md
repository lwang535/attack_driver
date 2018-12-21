
# Explore and visualization
### 正负样本比的统计

->>>情况：正负样本严重失衡，比例在7:93的样子

->>>解决方法：下采样
### 问题长度分布的统计

->>>情况：负样本是明显的长尾分布，99%的问题词数都少于30；正样本有点像均匀分布，99%的问题词数在5-45之间。

->>>思考：model1对每个问题仅取前30词是否会削弱对正样本的预测？

->>>解决方法：将词数放大到46或者50词
### 其他的feature engineering，比如tf-idf


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

2、data cleaning 步骤是否需要去掉stopwords啥的？

3、挖feature吗？

4、如果要增加运算量，比如增加batch size, epoch, 样本数量， 具体可以增加到多少？ kaggle的极限是？

5、GPU尝试？

6、cross validation 需要做吗？

7、对过拟合问题，需要尝试正则化吗？
同时使用了L1和L2正则，L1效果太强导致所有的权重收敛到0，最后所有的预测结构都会是0，所以最多使用L2权重进行正则化。
