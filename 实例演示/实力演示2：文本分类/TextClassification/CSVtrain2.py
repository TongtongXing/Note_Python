import pandas as pd
import time
import tensorflow
import lightgbm as lg
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm


train_data = pd.read_csv('mytrain.csv', lineterminator='\n')
test_data = pd.read_csv('testCSV.csv', lineterminator='\n')

'''---------------利用LabelEncoder对数据标签进行规格化处理----------------------'''


def encodeLabel(data):
    listLable=[]
    for lable in data['lable']:
        listLable.append(lable)
    '''到这里都是把lable整合到一起，下面是规格化处理'''
    le = LabelEncoder()
    resultLable=le.fit_transform(listLable)
    return resultLable

trainLable=encodeLabel(train_data)
testLable=encodeLabel(test_data)


'''---------------这里出来是所有review的集合----------------------'''


def getReview(data):
    listReview=[]
    le = LabelEncoder()
    for review in data['review']:
        listReview.append(review)
    return listReview

trainReview=getReview(train_data)
testReview=getReview(test_data)

#  这里出来是频数向量：
stoplist=['.', '?', '!', ':', '-', '+', '/', '"', ',', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
cv=CountVectorizer(stop_words=stoplist)
wordBag=trainReview+testReview
cv.fit(wordBag)
test_count = cv.transform(testReview)
testCount = test_count.toarray()
train_count = cv.transform(trainReview)
trainCount = train_count.toarray()


#  下面是非深度学习算法的评测代码：

#  评测函数：


def classificate(estimator, trainReview, trainLable, testReview, testLable):
    start = time.time()
    #模型训练,fit通常都是指模型的学习、训练过程
    print('训练:')
    model = estimator
    model.fit(trainReview, trainLable)
    print(model)
    #模型预测：
    print('预测:')
    pred_model = model.predict(testReview)
    print(pred_model)
    #算法评估
    print('评估:')
    score = metrics.accuracy_score(testLable, pred_model)
    matrix = metrics.confusion_matrix(testLable, pred_model)
    report = metrics.classification_report(testLable, pred_model)

    print('>>>准确率\n', score)
    print('\n>>>混淆矩阵\n', matrix)
    print('\n>>>召回率\n', report)
    end = time.time()
    t = end - start
    print('\n>>>算法消耗时间为：', t, '秒\n')


#  算法调用：
knc = KNeighborsClassifier()
classificate(knc, trainCount, trainLable, testCount, testLable)

# nb = GaussianNB()
# classificate(nb, trainCount, trainLable, testCount, testLable)


#  下面是前馈神经网络的评测代码：
'''
start = time.time()
feature_num = trainCount.shape[1]     # 设置所希望的特征数量

# 求标签的独热编码矩阵
trainCate = to_categorical(trainLable)
testCate = to_categorical(testLable)

# 1 创建神经网络
network = models.Sequential()

# 2 添加神经连接层
# 第一层必须有并且一定是 [输入层], 必选
network.add(layers.Dense(     # 添加带有 relu 激活函数的全连接层
                         units=100,
                         activation='relu',
                         input_shape=(feature_num, )
                         ))
# 介于第一层和最后一层之间的称为 [隐藏层]，可选
network.add(layers.Dense(     # 添加带有 relu 激活函数的全连接层
                         units=100,
                         activation='relu'
                         ))
# 最后一层必须有并且一定是 [输出层], 必选
network.add(layers.Dense(     # 添加带有 softmax 激活函数的全连接层
                         units=2,   #输出层感知器个数即标签维度/类数
                         activation='softmax'
                         ))
# 3 编译神经网络
network.compile(loss='categorical_crossentropy',  # 分类交叉熵损失函数
                optimizer='rmsprop',
                metrics=['accuracy']              # 准确率度量，也是选定evaluate的指标值为准确度
                )
# 4 开始训练神经网络
network.fit(trainCount,     # 训练集特征
            trainCate,        # 训练集标签
            epochs=5,          # 迭代次数
            batch_size=100,    # 指定进行梯度下降时每个batch包含的样本数
            validation_data=(testCount, testCate)  # 验证测试集数据
            )
#利用模型进行预测
pred = network.predict(testCount)
score = network.evaluate(testCount,
                        testCate,
                        batch_size=32)

end = time.time()
usingTime=end-start
print('时间:',usingTime,'s;   得分:',score)
'''

#下面是LSTM的评测代码：
'''
start = time.time()
feature_num = trainCount.shape[1]     # 设置所希望的特征数量

# 求标签的独热编码矩阵
trainCate = to_categorical(trainLable)
testCate = to_categorical(testLable)

# 1 创建神经网络
lstm_network = models.Sequential()

# 2 添加神经连接层
lstm_network.add(layers.Embedding(input_dim=feature_num,  # 添加嵌入层
                                  output_dim=28))

lstm_network.add(layers.LSTM(units=28))                 # 添加 28 个单元的 LSTM 神经层

lstm_network.add(layers.Dense(units=2,
                               activation='sigmoid'))     # 添加 sigmoid 分类激活函数的全连接层

# 3 编译神经网络
lstm_network.compile(loss='binary_crossentropy',
                optimizer='Adam',
                metrics=['accuracy']             # 准确率度量，也是选定evaluate的指标值为准确度
                )

# 4 开始训练神经网络
lstm_network.fit(trainCount,     # 训练集特征
            trainCate,        # 训练集标签
            epochs=5,          # 迭代次数
            batch_size=28,    # 指定进行梯度下降时每个batch包含的样本数
            validation_data=(testCount, testCate)  # 验证测试集数据
            )

#利用模型进行预测
pred = lstm_network.predict(testCount)
score = lstm_network.evaluate(testCount,
                        testCate,
                        batch_size=32)

end = time.time()
usingTime=end-start
print('时间:',usingTime,'s;   得分:',score)
'''