import os
import numpy as np
from PIL import Image
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, MaxPool2D, Flatten, Dense
from keras.optimizers import Adam
import shutil

# 文件重命名
File_path = "Raw_img"
Width = 100
Height = 100
Save_path = "Resize_img"
Nb_class = 4
Nb_batch_size = 64
Nb_epochs = 20


class PreImage(object):
    """
    File_path是未处理照片路径
    Save_path是调整大小后导出照片路径
    """
    def __init__(self, File_path, Save_path, Width=100, Height=100):
        self.path = File_path
        self.width = Width
        self.height = Height
        self.save_path = Save_path

    # 1图片重命名
    def ImageRename(self):
        File_sub_path_list = os.listdir(self.path)
        path_counter = 0
        for File_sub_path in File_sub_path_list:
            file_counter = 0
            for image in os.listdir(os.path.join(self.path, File_sub_path)): #1****************************
                os.rename(os.path.join(self.path, File_sub_path) + "/" + image,
                          os.path.join(self.path, File_sub_path) + "/" + str(path_counter) + "__" + str(file_counter) + ".jpg")
                # 如果出现这次创建的文件名和rename之前的文件名名字相同，则会报错，提示文件已经存在无法创建该文件。
                # os.rename(name1, name2)注意这个地方的name1是要到子root目录
                file_counter = file_counter + 1
            path_counter = path_counter + 1
        print(">>>>>>>重命名成功>>>>>>>>")

    # 2图片大小改变
    def ImageResize(self):
        # if os.path.exists(self.resized_path):
        #     shutil.rmtree(self.resized_path)  # 删除文件夹，无论空不空
        #     # 不适用先删除文件夹指令也行，因为每一次会覆盖上一次5****************************
        #     # os.remove(path)  # 删除文件
        #     # os.removedirs(path)  # 删除空文件夹
        #     # shutil.rmtree(path)  # 递归删除文件夹2****************************

        for root, dirs, files in os.walk(self.path):  # root会变化，变化成图片所在的根目录，dirs是当前目录下包括的文件夹
            for filename in files:
                # print(os.path.join(root, filename))
                image = Image.open(os.path.join(root, filename))  # 可以多路径join
                image_RGB = image.convert("RGB")
                image_Resize = image_RGB.resize((self.width, self.height), Image.BILINEAR)
                if not os.path.exists(self.save_path):
                    os.mkdir(self.save_path)
                # print(filename)
                image_Resize.save(os.path.join(self.save_path, os.path.basename(filename)))
                # 保存图片时，必须先对图片进行灰度化或者RGB化3****************************
        print(">>>>>>>图片大小更改成功,保存到当前%s文件夹>>>>>>>>" % self.save_path)


class Training(object):
    """
    File_path:待训练图片路径
    图片输入大小固定100*100,图片种类数量可变.
    """
    def __init__(self, Save_path, Nb_class, Nb_batch_size, Nb_epochs, Width=100, Height=100):
        self.path = Save_path
        self.nb_class = Nb_class
        self.nb_batch_size = Nb_batch_size
        self.nb_epochs = Nb_epochs
        self.width = Width
        self.height = Height

    # 1提取图片矩阵
    def Train(self):
        # 类里面函数之间的调用，也是需要self.Train(),也是需要使用self来实现的.
        Train_img_list = []
        Train_label_list = []
        for image in os.listdir(self.path):
            img = Image.open(os.path.join(self.path, image))
            image_array = np.array(img)
            Train_img_list.append(image_array)
            Train_label_list.append(int(image.split("_")[0]))  # 此处加上一个int
        Train_label_array = np.array(Train_label_list)  # 网络需要传入的是numpy数组而不是list
        Train_img_array = np.array(Train_img_list)
        print(Train_label_array.shape)
        print(Train_img_array.shape)
        # Train_img_array = Train_img_array.reshape(-1, self.width, self.height, 1)  # 这步骤多余.
        Train_img_array = Train_img_array / 255.
        print(type(Train_label_array))
        Train_label_array = np_utils.to_categorical(Train_label_array, 4)

        # IndexError: index 4 is out of bounds for axis 1 with size 4
        # 类标号要从0开始.所以图片rename的时候应该是0开头

        # 模型建立
        model = Sequential()
        model.add(Convolution2D(
            input_shape=(100, 100, 3),
            filters=32,
            kernel_size=(5, 5),
            padding="same"
        ))
        model.add(Activation("relu"))
        model.add(MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding="same"
        ))
        model.add(Convolution2D(
            filters=64,
            kernel_size=(5, 5),
            padding="same"
        ))
        model.add(Activation("relu"))
        model.add(MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding="same"
        ))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(Dense(256))
        model.add(Activation("relu"))
        model.add(Dense(self.nb_class))
        model.add(Activation("softmax"))

        # 模型编译
        adam = Adam(lr=0.001)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=adam,
            metrics=["accuracy"]
        )

        # 模型启动
        model.fit(
            x=Train_img_array,
            y=Train_label_array,
            batch_size=self.nb_batch_size,
            epochs=self.nb_epochs,
            verbose=1
        )
        model.save("./dog.h5")


if __name__ == "__main__":

    """第一个类:预处理图片"""
    DogProcess = PreImage(File_path, Save_path, Width, Height)
    # # 图片重命名
    DogProcess.ImageRename()
    # # 图片大小尺寸改变
    DogProcess.ImageResize()

    """第二个类:训练模型"""
    DogTraining = Training(Save_path, Nb_class, Nb_batch_size, Nb_epochs)
    DogTraining.Train()
