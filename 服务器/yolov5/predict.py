
import io
import numpy as np
import cv2
import torch
from PIL import Image
from numpy import random
'''
代码：由YOLOv5自带的detect.py 改编

'''

from utils.plots import Annotator, colors, save_one_box
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression,   set_logging

from utils.torch_utils import select_device
from utils.plots import Colors

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    将预测的坐标信息coords(相对img1_shape)转换回相对原图尺度（img0_shape）
    :param img1_shape: 缩放后的图像大小  [H, W]=[384, 512]
    :param coords: 预测的box信息 [7,4]  [anchor_nums, x1y1x2y2] 这个预测信息是相对缩放后的图像尺寸（img1_shape）的
    :param img0_shape: 原图的大小  [H, W, C]=[375, 500, 3]
    :param ratio_pad: 缩放过程中的缩放比例以及pad  一般不传入
    :return: coords: 相对原图尺寸（img0_shape）的预测信息
    """
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        # gain = old/new = 1.024  max(img1_shape): 求img1的较长边  这一步对应的是之前的letterbox步骤
        gain = max(img1_shape) / max(img0_shape)
        # wh padding 这一步起不起作用，完全取决于letterbox的方式
        # 当letterbox为letter_pad_img时，pad=(0.0, 64.0); 当letterbox为leeter_img时,pad=(0.0, 0.0)
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    # 将相对img1的预测信息缩放得到相对原图img0的预测信息
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain        # 缩放
    # 缩放到原图的预测结果，并对预测值进行了一定的约束，防止预测结果超出图像的尺寸
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, img_shape):
    """
    Clip bounding xyxy bounding boxes to image shape (height, width)
    c.clamp_(a, b): 将矩阵c中所有的元素约束在[a, b]中间
                    如果某个元素小于a,就将这个元素变为a;如果元素大于b,就将这个元素变为b
    这里将预测得到的xyxy做个约束，是因为当物体处于图片边缘的时候，预测值是有可能超过图片大小的
    :param boxes: 函数开始=>缩放到原图的预测结果[7, 4]
                  函数结束=>缩放到原图的预测结果，并对预测值进行了一定的约束，防止预测结果超出图像的尺寸
    :param img_shape: 原图的shape [H, W, C]=[375, 500, 3]
    """
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


weights = '/root/yolov5/yolov5s.pt'  # 训练好的模型位置
opt_device = '0'  # device = 'cpu' or '0' or '0,1,2,3'
imgsz = 640
opt_conf_thres = 0.25
opt_iou_thres = 0.5

# Initialize
set_logging()
device = select_device(opt_device)
half = device.type != 'cpu'  # half precision only supported on CUDA

# 加载模型
model = attempt_load(weights, device=device)  # load FP32 model
imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
if half:
    model.half()  # to FP16

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names  # 获取标签
print(names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]


# def transform_image(image_bytes):
#     image = Image.open(io.BytesIO(image_bytes))
#     # img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
#     # img = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR)
#     # print(img)
#     return image


# （接口中传输的二进制流）将二进制用cv2 读取流并转换成yolov5 可接受的图片
def bytes_img(image_bytes):
    # 二进制数据流转np.ndarray [np.uint8: 8位像素]
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb_img

# '''
# [[[ 60  65  68]
#   [202 205 213]
#   [207 208 228]
# '''

from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression
# from utils.datasets import letterbox
# from utils.plots import plot_one_box





colors = Colors()  # create instance for 'from utils.plots import colors'

# 此函数用来接收numpy格式的图片，进行检测并返回目标信息（类别，位置）
def predict_(pic_):
    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    # Set Dataloader & Run inference
    im0s = pic_  # BGR  # 蓝绿红
    img = letterbox(im0s, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    # pred = model(img, augment=opt.augment)[0]
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt_conf_thres, opt_iou_thres)
    # print(type(pred))
    # Process detections
    detect_info = []
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                # print(cls)
                dic ={
                    'class':f'{names[int(cls)]}', # 检测目标对应的类别名
                    'location':torch.tensor(xyxy).view(1, 4).view(-1).tolist(), # 坐标信息,左上和右下的坐标
                    'score': round(float(conf) * 100, 2)  # 目标检测分数
                }
                detect_info.append(dic)

                #画框
                c = int(cls)
                color = colors(c, True)
                p1,p2=(int(xyxy[0]),int(xyxy[1])),(int(xyxy[2]),int(xyxy[3]))

                img=cv2.rectangle(img=pic_,pt1=p1,pt2=p2,color=color,thickness=1)
                img=cv2.putText(img,str(names[c]),(p1[0], p1[1] - 2),0,0.35,color,1)
            # cv2.imshow('frame',img)
            # cv2.waitKey(1)

    return detect_info

# img=cv2.imread(r'E:\01_school_study\01_python_project\yolo_project\traffic_sign\data\images\test\1_0.jpg')
# predict_(img)
# if __name__=='__main__':
#     img=cv2.imread('data/images/test/1_0.jpg')
#     print(predict_(img))
