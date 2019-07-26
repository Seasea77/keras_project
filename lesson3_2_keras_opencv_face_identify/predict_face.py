from keras.models import load_model
import cv2
import numpy as np
from PIL import Image


model = load_model("./face.h5")
# 先加载模型，避免在每次循环中再加载模型，这样会非常慢。
FACE_MODEL = cv2.CascadeClassifier("D:\\software-installation\\opencv\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml")


class Predict_face():
    def __init__(self):
        pass

    def process_image(self, img):
        # img = Image.open(img)
        img = img.convert("L")
        img = img.resize((100, 100), Image.BILINEAR)
        img = np.array(img)
        img = img.reshape(1, 100, 100, 1)
        img = img / 255.
        return img

    def predict_img(self, image):
        """此处的image为摄像头获取到的图片"""
        img = self.process_image(image)
        prediction = model.predict(img)
        final_prediction = np.argmax(prediction)
        print(final_prediction)
        name = ["WHD", "WLS"]
        for i in range(2):
            if final_prediction == i:
                result_name = name[i]
        return result_name

    def detect_face(self):
        capture = cv2.VideoCapture(0)
        COUNTER = 0
        while(True):
            ret, frame = capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = FACE_MODEL.detectMultiScale(gray)
            for (x, y, w, h) in faces:
                COUNTER += 1
                frame_img = Image.fromarray(frame)
                frame_img = frame_img.crop((x - 40, y - 40, x + w + 40, y + h + 40))  # 这个裁剪只适用于img图片，而不是frame矩阵
                predict_result = self.predict_img(frame_img)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)
                cv2.putText(frame, predict_result, (x, y - 40), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0), thickness=1)
            # cv2.namedWindow("人脸识别", cv2.WINDOW_AUTOSIZE)
            # cv2.imshow("人脸识别", frame)
            cv2.namedWindow("人脸识别", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("人脸识别".encode("gbk"), frame)
            if cv2.waitKey(1) & 0xFF == "x":
                break
        capture.release()
        cv2.destroyAllWindows()


predict_face = Predict_face()
predict_face.detect_face()
