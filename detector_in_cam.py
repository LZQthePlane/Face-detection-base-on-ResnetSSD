from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import time
import os
import cv2 as cv


file_path = os.path.dirname(os.path.abspath(__file__)) + os.sep
threshold = 0.5  # objects' confidence threshold

# 加载预训练后的ResNetSSD的caffe模型
prototxt_file = file_path + 'Resnet_SSD_deploy.prototxt'
caffemodel_file = file_path + 'Res10_300x300_SSD_iter_140000.caffemodel'
net = cv.dnn.readNetFromCaffe(prototxt_file, caffeModel=caffemodel_file)
print('ResNetSSD caffe model loaded successfully')

# 获取摄像头
# 这里使用的是opencv的API，而非imutils中的VideoStream，cap.read()返回值有所不同
cap = cv.VideoCapture(0)
time.sleep(1.0)
fps = FPS().start()

# 输出视频的相关参数
size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
out_fps = 20  # 输出视频的帧数
fourcc = cv.VideoWriter_fourcc(*'mp4v')  # 输出视频的格式
writer = cv.VideoWriter()
out_path = file_path+'test_out'+os.sep+'example.mp4'
writer.open(out_path, fourcc, out_fps, size, True)

while True:
    _, frame = cap.read()
    origin_h, origin_w = frame.shape[:2]
    blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            bounding_box = detections[0, 0, i, 3:7] * np.array([origin_w, origin_h, origin_w, origin_h])
            x_start, y_start, x_end, y_end = bounding_box.astype('int')

            # 显示image中的object类别及其置信度
            label = '{0:.2f}%'.format(confidence * 100)
            # 画bounding box
            cv.rectangle(frame, (x_start, y_start), (x_end, y_end),(0, 0, 255), 2)
            # 画文字的填充矿底色
            cv.rectangle(frame, (x_start, y_start-18), (x_end, y_start), (0, 0, 255), -1)
            # detection result的文字显示
            cv.putText(frame, label, (x_start+2, y_start-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    fps.update()
    fps.stop()
    text = "FPS: {:.2f}".format(fps.fps())
    cv.putText(frame, text, (15, int(origin_h * 0.92)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv.imshow('Frame', frame)
    writer.write(frame)

    if cv.waitKey(1) & 0xFF == ord("q"):  # 退出键
        break

# print('Elapsed time: {0:.2f}'.format(fps_imutils.elapsed()))  # webcam运行的时间
# print('Approximate FPS: {0:.2f}'.format(fps_imutils.fps()))  # 每秒帧数
writer.release()
cap.release()
cv.destroyAllWindows()