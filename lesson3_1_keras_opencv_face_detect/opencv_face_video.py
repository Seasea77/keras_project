"""1视频检测人脸"""


import cv2


# 导入人脸检测模型
FaceModel = cv2.CascadeClassifier("D:\\software-installation\\opencv\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml")
# FaceModel = cv2.CascadeClassifier("D:\\software-installation\\opencv\\opencv\\build\\etc\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml")
# 替换成绝对路径，并且用\\双反斜杠

# 开始摄像头
capture = cv2.VideoCapture(0)
# 若有多个摄像头，从0,1,2开始读摄像头。读视频只需要把0换成视频路径加名称。
# opencv读过来的视频是没有声音的。


while True:
    # 读取摄像头数据
    ret, frame = capture.read()  # 不要写反了*********************************1
    # 返回两个值，ret表示读取图像True，frame表示一帧一帧的图

    # 降维打击，3维变成1维，运算轻松很多。
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_RGB2GRAY)

    # 检测人脸
    faces = FaceModel.detectMultiScale(gray, scaleFactor=1.1)
    print("faces:", faces)
    # 影响整体检测，不能低于1.1，到3左右
    # faces返回的是人脸坐标

    # 标记人脸
    for (x, y, w, h) in faces:  # faces不能是列表，是二维矩阵。
        print(x, y, w, h)
        print("_______________")
        # 画外边框
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        # 添加名字
        cv2.putText(frame, "HD", org=(x, y-10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0), thickness=1)

        # 创建窗口
        cv2.namedWindow("LiveFace", cv2.WINDOW_AUTOSIZE)  # 窗口大小不可改变
        # cv2.namedWindow("input image", 1)  # 1为自动调整窗口大小模式, cv2.namedWindow("input image", 0)  #  0窗口大小可以调整
        # cv2.WINDOW_FREERATIO  自适应比例

        # 放窗口里面
        cv2.imshow("LiveFace", frame)
        if cv2.waitKey(1) & 0xFF == ord("x"):  # 在显示窗口上按x结束窗口显示。
            break
        # cv2.waitKey(0)  # 参数none和0是代表无限延迟，而整数数字代表延迟多少ms。
        # cv2.waitKey(1)  # 最高速率取照片。如果把图片关掉，则该指令结束。
capture.release()  # 释放资源
cv2.destroyAllWindows()  # 关闭窗口