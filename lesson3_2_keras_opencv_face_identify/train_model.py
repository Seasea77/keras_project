import os
import numpy as np
from PIL import Image
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, MaxPool2D, Flatten, Dense, Dropout
from keras.optimizers import Adam


class Train():
    def __init__(self, Raw_path, batch_size=20, nb_epoch=10):
        self.raw_path = Raw_path
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch

    def pre_img(self):
        img_array_list = []
        label_array_list = []
        nb_class = 0
        for _ in os.listdir(self.raw_path):
            nb_class += 1
        for root, dirs, files in os.walk(self.raw_path):
            for filename in files:
                img = Image.open(os.path.join(root, filename))
                img = img.convert("L")
                img = img.resize((100, 100), Image.ANTIALIAS)
                img_array = np.array(img)
                img_array = img_array.reshape(100, 100, 1)
                img_array = img_array / 255.
                img_array_list.append(img_array)
                label_array_list.append(int(filename.split("_")[0]))
        img_array = np.array(img_array_list)
        label_array = np.array(label_array_list)
        # print(max(label_array))  # 找最大的数，这个地方不是np.argmax而是max
        label_array = np_utils.to_categorical(label_array, nb_class)
        print(img_array.shape)
        print(label_array)
        return img_array, label_array

    def train_model(self):
        img_array, label_array =self.pre_img()

        model = Sequential()

        model.add(Convolution2D(
            input_shape=(100, 100, 1),
            filters=32,
            kernel_size=(3, 3),
            padding="same",
        ))
        model.add(Activation("relu"))
        model.add(MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding="same"
        ))

        model.add(Convolution2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
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
        model.add(Dropout(0.2))
        model.add(Dense(256))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))
        model.add(Dense(2))
        model.add(Activation("softmax"))

        adam = Adam(lr=0.001)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=adam,
            metrics=["accuracy"]
        )

        model.fit(
            x=img_array,
            y=label_array,
            batch_size=self.batch_size,
            epochs=self.nb_epoch,
            verbose=1
        )
        model.save("./face.h5")


def main():
    Train_model = Train("Face_image")
    # Train_model.pre_img()
    Train_model.train_model()


if __name__ == "__main__":
    main()