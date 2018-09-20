import numpy as np
import cv2 as cv
import os


file_path = os.path.dirname(os.path.abspath(__file__)) + os.sep
threshold = 0.5  # human face's confidence threshold

# 加载预训练后的ResNetSSD的caffe模型
prototxt_file = file_path + 'Resnet_SSD_deploy.prototxt'
caffemodel_file = file_path + 'Res10_300x300_SSD_iter_140000.caffemodel'
net = cv.dnn.readNetFromCaffe(prototxt_file, caffeModel=caffemodel_file)
print('MobileNetSSD caffe model loaded successfully')

# 读取图片
image = cv.imread(file_path + 'test' + os.sep + 'example_02.jpg')
origin_h, origin_w = image.shape[:2]

# 对图像进行预处理：进行resize、mean_subtrction以及scale
# https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# 将blob作为输入传入网络中，并进行正向传播以得到输出
net.setInput(blob)
# detections是一个4维列表
# 第三维是image上检测出的face的个数
# 第四维的1是object的类别号，2是置信度，3:7是bounding box位置值
detections = net.forward()
print('Face detection accomplished')

# 遍历每一个face
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > threshold:
        # 取出bounding box的位置值并还原到原始image中
        bounding_box = detections[0, 0, i, 3:7] * np.array([origin_w, origin_h, origin_w, origin_h])
        x_start, y_start, x_end, y_end = bounding_box.astype('int')

        label = '{0:.2f}%'.format(confidence * 100)
        cv.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
        # 画文字的填充矿底色
        cv.rectangle(image, (x_start, y_start - 18), (x_end, y_start), (0, 0, 255), -1)
        cv.putText(image, label, (x_start+2, y_start-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cv.imshow('output', image)
cv.waitKey(0)
cv.destroyAllWindows()