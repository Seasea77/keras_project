"""2静态图片检测"""


import cv2

FaceModel = cv2.CascadeClassifier("D:\\software-installation\\opencv\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml")
frame = cv2.imread("test.jpg")
print(frame)
gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
faces = FaceModel.detectMultiScale(gray, scaleFactor=1.2)
# 如果检测出有n张脸，则faces有n行数据 *********************************2
# [[ 634  272  170  170]
#  [ 267  341  295  295]
#  [ 975  393  392  392]
#  [ 108  658  281  281]
#  [1291   27   86   86]
#  [ 711   69  178  178]
#  [ 773  599  272  272]
#  [ 597  587   54   54]]
for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
    cv2.putText(frame, "HDHD", org=(x, y-10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0), thickness=1)
    cv2.namedWindow("Face_detection", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Face_detection", frame)
cv2.waitKey(0)  # 由于只有一张图片
cv2.destroyAllWindows()