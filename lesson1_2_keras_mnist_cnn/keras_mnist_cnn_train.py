# coding:utf-8


import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils  # 统一处理numpy数据的工具
from keras.layers import Convolution2D, Activation, MaxPool2D, Flatten, Dense, Dropout
# MaxPool2D和MaxPooling2D的区别??
from keras.optimizers import Adam
from keras.models import Sequential


nb_class = 10
nb_epoch = 2
batch_size = 1024  # 128,内存小改为64，内存大改大些,跑起来速度变化不大就行。
# 这个参数有时很有个性,需要调节试试,有时loss很快,改一下有时loss就会很小
# 调参还可以调节lr学习率.


"""1数据读取与处理"""
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1)    # channel_last通道数在最后一个维度；-1看成未知
X_test = X_test.reshape(-1, 28, 28, 1)    # channel_last通道数在最后一个维度；-1看成未知

# # 使用下面两条语句查看X_train是否归一化处理了
# a_csv = X_train.reshape(-1, 784)
# np.savetxt("a.csv", a_csv, delimiter=",")  # 1个多G。
# 实际是没有转

X_train = X_train / 255.  # 必须是浮点数，不然很难收敛。
Y_train = np_utils.to_categorical(Y_train, nb_class)
Y_test = np_utils.to_categorical(Y_test, nb_class)


"""2网络搭建（卷积网络+全连接网络）"""
model = Sequential()

# 1st Conv2D layer
model.add(Convolution2D(
    filters=32,
    kernel_size=(5, 5),
    padding="same",
    input_shape=(28, 28, 1),
))

model.add(Activation("relu"))

model.add(MaxPool2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding="same",
))

# 2nd Conv2D layer
model.add(Convolution2D(
    filters=64,
    kernel_size=(5, 5),
    padding="same"
))

model.add(Activation("relu"))

model.add(MaxPool2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding="same",
))
# 卷积完成后生成的形状为[[[1,2,3],[2,3,4][4,5,6]]]，在此项目中，因为是灰度图，所以是3维，一般是4维
# 根据实际情况加卷积层数。

# 1st Fully connected layer
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
# model.add(Dropout(0.2))

# 2nd Fully connected layer
model.add(Dense(nb_class))
model.add(Activation("softmax"))


"""3编译"""
adam = Adam(lr=0.01)  # 实例化, 0.01根据实际情况来
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)


"""4启动网络-训练网络"""
model.fit(
    x=X_train,
    y=Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,  # 输入一个字母，有等号的提示是需要参数(最前面有个小提示P)，
    # 没等号的提示是已经定义好的变量（最前面有个小提示V）。
    verbose=1,
    validation_data=[X_test, Y_test],
)
model.save("./conv_model.h5")  # 模型的保存
# 旧版本中写下面实现上面validation_data=[X_test, Y_test]的功能
# evaluation = model.evaluate(X_test, Y_test)
# print(evaluation)

