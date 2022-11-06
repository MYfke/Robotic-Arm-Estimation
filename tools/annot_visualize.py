import cv2
import json
from easydict import EasyDict

cv2.namedWindow("Image")  # 创建窗口
with open('..\\data\\RoboArm\\annot\\train.json') as label:
    for item in json.load(label):
        item = EasyDict(item)
        print(item.imgname)
        img = cv2.imread('..\\data\\RoboArm\\images\\' + item.imgname)  # 读取图像
        pt1 = (int(item.bbox_center[0] - item.bbox_size[0]/2), int(item.bbox_center[1] - item.bbox_size[1]/2))
        pt2 = (int(item.bbox_center[0] + item.bbox_size[0]/2), int(item.bbox_center[1] + item.bbox_size[1]/2))
        cv2.rectangle(img, pt1, pt2, color=(255, 0, 0),thickness=5)
        for point in item.keypoints:
            # noinspection PyTypeChecker
            a = (int(point['coord'][0]), int(point['coord'][1]))
            cv2.circle(img, a, 4, (255, 255, 0), -1)  # 画圆形
        cv2.imshow('Image', cv2.resize(img, None, fx=0.3, fy=0.3))
        if cv2.waitKey(200) == 27:
            break
cv2.destroyWindow("Image")  # 关闭窗口
