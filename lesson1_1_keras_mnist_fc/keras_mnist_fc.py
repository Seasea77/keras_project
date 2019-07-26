# coding:utf-8
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Dropout
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
from PIL import Image


"""1设置基本参数"""
batch_size = 1024  # 一次训练需要给神经网络注入多少个数据，即神经网络跑一次，能跑1024个数据。这个值取得太大效果不好
num_class = 10  # 训练分类的个数
num_epochs = 2  # 训练的次数（所有60000个数据用一次叫做1个epoch）


"""2加载原始数据"""
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()  # 加载进来的二值图就是矩阵
# （1）mnist数据集的加载
# ——问题：
# # 使用(X_train, Y_train), (X_test, Y_test) = mnist.load_data()下载失败，所以无法加载成功。
# ——解决：
# # 加载失败会提示下载地址，通过下载，通过翻墙，下载速度很快。
# # 在执行上述指令时，mnist.load_data()，会在home(C:/Users/61052)目下的.keras目录下生成datasets文件夹，
# 因为下载失败，所以文件夹为空。
# # cmd打开终端，在home目录下，cd .keras、cd datasets、start .（打开当前文件夹），执行将下载好的mnist
# 数据集放入到文件夹再运行就不会报错。
#
# # X_train.shape=(60000, 28, 28)
# Y_train.shape=(60000,)
# X_test.shape=(10000, 28, 28)
# Y_test.shape=(10000,)
# # 加载进来的二值图就是矩阵


"""3处理数据集，归一化"""
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# 转换成小数，进行归一化(相当于是BN)
# X_train = X_train.astype("float32")
X_train = X_train / 255.  # 255加小数点；print(X_train[9999]) #打印其中任意一行，看一下

# 将整型标签转为onehot
Y_train = np_utils.to_categorical(Y_train, num_class)  # Y_train必须为数组
Y_test = np_utils.to_categorical(Y_test, num_class)  # 可以print(Y_test[99])看其中一个标签的形式


"""4设置网络结构"""
model = Sequential()  # 开始keras的序列模型

# 第一层
model.add(Dense(512, input_shape=(784,)))  # 第一层必须告诉系统你的输入是多少，默认1行784列，输入是(*,784),输出是(*,512)
# (784, )逗号不能少，(784,1)或者写成这样
model.add(Activation("relu"))
model.add(Dropout(0.2))  # 随机失活一部分

# 第二层
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.2))

# 第三层
model.add(Dense(10))
model.add(Activation("softmax"))  # softmax针对10个进行分类


"""5编译"""
# 用什么函数来处理
model.compile(
    loss="categorical_crossentropy",  # 损失函数
    optimizer="rmsprop",  # 优化函数，SGD、adam不同优化函数不同的路径向最优点推进，adam最新
    metrics=["accuracy"]  # 达到的目标
)
# model.save("./mine.h5")  # 模型的保存


"""6启动网络--训练网络"""
Trainning = model.fit(
    X_train, Y_train,
    batch_size=batch_size,
    epochs=num_epochs,
    validation_data=(X_test, Y_test),  # 可加可不加，输出多一条数据
    # - 3s - loss: 0.5421 - acc: 0.8313 - val_loss: 1.2277 - val_acc: 0.9230
    verbose=2
    # 默认为1，输出显示进度条，显示每个Epoch
    # 2，输出不显示进度条，显示每个Epoch
    # 0，输出什么都不显示
)


"""7把Training实例化"""
print("_________________________________________________________________")
print(Trainning.history)  #
print(Trainning.params)  # 网络的设置


"""8测试test数据"""
testpic = X_test[9998].reshape(1, 784)
testlabel = np.argmax(Y_test[9998])
testpic = testpic.reshape(28, 28)
print("标签结果：", testlabel)
plt.imshow(testpic)
plt.show()

# 模型预测结果
testpic = testpic.reshape(1, 784)
pred = model.predict(testpic)
print("预测结果：", np.argmax(pred))


"""9测试自己的照片"""

# # 方法1：变成灰度图
# def rgb2gray(rgb):
#     return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])
# mine_img = mpimage.imread("./test.jpg")
# mine_img_L = rgb2gray(mine_img)
# print(mine_img_L.shape)

# 方法2：变成灰度图
mine_img = Image.open("./test.png")  # open和imread的区别
# mine_img = Image.open("./1.png")  # open和imread的区别
mine_img_L = mine_img.convert("L")
mine_img_L = mine_img_L.resize((28, 28), Image.ANTIALIAS)   # 调整图片大小为(28, 28)
mine_img_L = np.array(mine_img_L)  # mine_img_L的shape为(28, 28)
# np.savetxt('4.csv', mine_img_L, delimiter=',')  # 把矩阵存放到csv文件里面
plt.imshow(mine_img_L)  # mine_img_L.show(),使用电脑自带的照片显示软件。plt.imshow(mine_img_L),使用plt来显示图片。
plt.show()
print(mine_img_L.shape)

# 模型预测结果
mine_img_L = mine_img_L.reshape(1, 784)
mine_img_L = mine_img_L / 255.
pred = model.predict(mine_img_L)
print(np.argmax(pred))
