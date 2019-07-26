import os
import cv2
import random
from PIL import Image
from time import sleep


class Collect_image(object):
    def __init__(self):
        pass

    # 为新用户创建文件夹
    def mkdir(self, Face_folder, name):
        if not os.path.exists(Face_folder):
            os.mkdir(Face_folder)
        Face_sub_folder_list = os.listdir(Face_folder)
        if Face_sub_folder_list == []:
            os.mkdir(os.path.join(Face_folder, "0_" + name))
            name_path = os.path.join(Face_folder, "0_" + name)
            folder_num = 0
        else:
            name_num = []
            for sub_name in Face_sub_folder_list:
                name_num.append(int(sub_name.split("_")[0]))  # 此处必须要加一个int,否则出来的就是str,str会导致"10"排在9的前面
            name_num.sort()
            os.mkdir(os.path.join(Face_folder, str(name_num[-1] + 1) + "_" + name))
            folder_num = name_num[-1] + 1
            name_path = os.path.join(Face_folder, str(name_num[-1] + 1) + "_" + name)
        return folder_num, name_path

    def collect_image(self, Face_folder, collect_face_num):
        # 1创建用户文件夹
        name = input("请输入名字-->")
        folder_num, name_path = self.mkdir(Face_folder, name)

        # 2获取用户图片
        FACE_DETECTOR = cv2.CascadeClassifier("D:\\software-installation\\opencv\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml")

        capture = cv2.VideoCapture(0)
        face_counter = 0
        sleep(2)  # opencv在调用摄像头时会有启动时间
        while(True):
            print(face_counter)
            ret, frame = capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # COLOR_RGB2GRAY不要写错了 写成COLOR_RGB2GBGRA就会报错
            faces = FACE_DETECTOR.detectMultiScale(gray, scaleFactor=1.2)
            print(faces)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
                if face_counter < collect_face_num:
                    cv2.putText(img=frame,
                                text="collect%simage" % str(face_counter),
                                org=(x, y - 20),
                                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                fontScale=1,
                                color=(0, 255, 0),
                                thickness=1)
                    random_num = random.randint(100000, 999999)
                    gray_img = Image.fromarray(gray)  # ***********************************1
                    gray_img.save(os.path.join(name_path, str(folder_num) + "_" + name + "_" + str(random_num) + ".jpg"))
                    gray_face = gray_img.crop((x - 40, y - 40, x + w + 40, y + h + 40))
                    # （1）4个点的坐标，（2）且4个点坐标单独加括号***********************************2
                    # crop_img = img.crop((x-40, y-40), (x+w+40, y+h+40))  # 只能接受一个参数
                    gray_face.save(os.path.join(name_path, str(folder_num) + "_" + name + "_" + str(random_num) + ".jpg"))

                else:
                    cv2.putText(img=frame,
                                text="images have collected ",
                                org=(x, y - 20),
                                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                fontScale=1,
                                color=(0, 255, 0),
                                thickness=1)
                    # cv2.putText() 只能显示英文字符******************************3
                    # # 想要显示中文字符，用下面的方法********************************4
                    # # 这个是在图片上显示中文，不是在frame上面显示中文。
                    # frame = Image.fromarray(frame)
                    # draw = ImageDraw.Draw(frame)
                    # font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")
                    # draw.text((x, y - 10), text="我的图片", fill=(0, 0, 255), font=font)
                face_counter += 1
            cv2.namedWindow("my_face", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("my_face", frame)
            if face_counter > collect_face_num:
                break
            if cv2.waitKey(500) & 0xFF == "x":
                break


def main():
    Collect_img = Collect_image()
    Collect_img.collect_image("Face_image", 30)


if __name__ == "__main__":
    main()
