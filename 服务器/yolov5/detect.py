# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
from flask import Flask, Response

app = Flask(__name__)
import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()                    #得到detect文件的绝对路径
ROOT = FILE.parents[0]  # YOLOv5 root directory   即detect的父目录，YOLOv5项目的根目录
if str(ROOT) not in sys.path:                     #判断YOLOv5是否在模块的查询列表中
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative  将root目录转变为相对路径

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(1024, 1024),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):

    source = str(source)            #确保路径是字符串类型
    save_img = not nosave and not source.endswith('.txt')  # save inference images    判断传入的路径文件（即source）是不是txt文件，如果不是则保存预测图片
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)        #判断路径文件的后缀是否在img和vid的后缀列表中
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))         #判断路径是不是网络流
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)     #判断传入的路径是否代表摄像头（0、1、2、3）或网络流；即判断是否为摄像头地址
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:          #判断是不是网络上的图片或视频流，如果是就去下载
        source = check_file(source)  # download

    # Directories 保存预测结果的路径
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run  实现每次运行该文件时，保存的文件的名字呈现增量变大，即runs/detect目录下的exp、exp2、exp3
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)       #加载使用gpu还是cpu
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)    #DetectMultiBackend代表多后端，即支持多种深度学习框架，如pytorch、tensorflow等，在函数内部判别当前使用的是哪种框架，从而使用该框架的方式加载模型权重
    stride, names, pt = model.stride, model.names, model.pt      #加载出模型的一些参数
    imgsz = check_img_size(imgsz, s=stride)  # check image size 确保图像的大小是模型的步长的倍数

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        #view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup 热身：让cpu或gpu先跑一张图片、热身一下
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    count = 0
    import time
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
        count += 1

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)   #包含4个坐标信息、一个置信度阈值信息和80个类别的概率值

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)   #预测出来的信息：eg、[1,5,6]，代表一个batch，五个检测框，每个检测框有六个信息：四个坐标、一个概率、一个类别
                                                                                                              #max_det代表一张图片中最多预测出多少个目标
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1    #计数
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string   !!!!此处信息举例：0: 480x640 8 persons, 1 wine glass, 1 cup, 1 tv, 1 laptop, 171.3ms
                                                                                    #可以通过该变量查询出是否有陌生人、狼，如果有就警报
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file 此处将检测的结果即类别、方框坐标和置信度写到文件中
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            print(txt_path)

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')     #决定是否画标签、置信度
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()            #得到画好框的图片

            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                #cv2.imshow(str(p), im0)          #处理后的视频流和图片在此处展示，可以从im0获取当前处理后的照片
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)           #如果保存，则处理后的图片会保存到该路径
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default='http://119.3.225.185:9000/?action=stream', help='file/dir/URL/glob/screen/0(webcam)')   #树莓派通过frp到服务器：
    #parser.add_argument('--fps', type=int, default=10, help='frames per second')#自己加的
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')    #三个--后面跟着的参数是一样的效果，有一个即可，默认图像大小为640*640
    parser.add_argument('--conf-thres', type=float, default=0.55, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand  一个很简洁、简单的应用，如果我执行命令时：python detect.py --imgsz 320，或者我执行命令时不指定图像大小，那么图像的大小列表只有一个值，而图像的大小是两维的，故将列表扩展为Y*Y
    print_args(vars(opt))
    return opt   #返回配置信息

@app.route('/processed_video_feed')
def processed_video_feed():
    #opt = parse_opt()
    #check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))  # 检测requirements文件中的包有没有安装成功

    def generate(
            weights=ROOT / 'yolov5s.pt',  # model path or triton URL
            source='http://60.204.139.74:9000/?action=stream',  # file/dir/URL/glob/screen/0(webcam)  http://60.204.218.251/?action=stream
            data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
            imgsz=(620, 480),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=True,  # update all models
            project=ROOT / 'runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=2,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=True,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            vid_stride=8,  # video frame-rate stride
    ):
        
        source = str(source)  # 确保路径是字符串类型
        save_img = not nosave and not source.endswith(
            '.txt')  # save inference images    判断传入的路径文件（即source）是不是txt文件，如果不是则保存预测图片
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)  # 判断路径文件的后缀是否在img和vid的后缀列表中
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))  # 判断路径是不是网络流
        webcam = source.isnumeric() or source.endswith('.streams') or (
                    is_url and not is_file)  # 判断传入的路径是否代表摄像头（0、1、2、3）或网络流；即判断是否为摄像头地址
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:  # 判断是不是网络上的图片或视频流，如果是就去下载
            # source = check_file(source)  # download
            source =  cv2.VideoCapture(source)

        # Directories 保存预测结果的路径
        save_dir = increment_path(Path(project) / name,
                                   exist_ok=exist_ok)  # increment run  实现每次运行该文件时，保存的文件的名字呈现增量变大，即runs/detect目录下的exp、exp2、exp3
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load modelruhe
        device = select_device(device)  # 加载使用gpu还是cpu
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data,
                                   fp16=half)  # DetectMultiBackend代表多后端，即支持多种深度学习框架，如pytorch、tensorflow等，在函数内部判别当前使用的是哪种框架，从而使用该框架的方式加载模型权重
        stride, names, pt = model.stride, model.names, model.pt  # 加载出模型的一些参数
        imgsz = check_img_size(imgsz, s=stride)  # check image size 确保图像的大小是模型的步长的倍数

        # Dataloader
        bs = 1  # batch_size
        if webcam:
            # view_img = check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup 热身：让cpu或gpu先跑一张图片、热身一下
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        count = 0
        import time
        num=0
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
            count += 1

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)  # 包含4个坐标信息、一个置信度阈值信息和80个类别的概率值

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms,
                                           max_det=max_det)  # 预测出来的信息：eg、[1,5,6]，代表一个batch，五个检测框，每个检测框有六个信息：四个坐标、一个概率、一个类别
                # max_det代表一张图片中最多预测出多少个目标
            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            num+=1
            for i, det in enumerate(pred):  # per image
                
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                if num%5==0:
                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # im.jpg
                    txt_path = str(save_dir / 'labels' / p.stem) + (
                        '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                    s += '%gx%g ' % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    imc = im0.copy() if save_crop else im0  # for save_crop
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
    
                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string   !!!!此处信息举例：0: 480x640 8 persons, 1 wine glass, 1 cup, 1 tv, 1 laptop, 171.3ms
                            # 可以通过该变量查询出是否有陌生人、狼，如果有就警报
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file 此处将检测的结果即类别、方框坐标和置信度写到文件中
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                with open(f'{txt_path}.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
                                    print(txt_path)
    
                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (
                                    names[c] if hide_conf else f'{names[c]} {conf:.2f}')  # 决定是否画标签、置信度
                                annotator.box_label(xyxy, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
    
                    # Stream results
                    im0 = annotator.result()  # 得到画好框的图片
                    #cv2.imshow(str(p), im0)
                    if view_img:
    
                        if platform.system() == 'Linux' and p not in windows:
                            windows.append(p)
                            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                        cv2.imshow(str(p), im0)          #处理后的视频流和图片在此处展示，可以从im0获取当前处理后的照片
                        cv2.waitKey(1)  # 1 millisecond                

                ret, im0 = cv2.imencode('.jpg', im0)
                
                LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + im0.tobytes() + b'\r\n')
            # 设置 Response 的 MIME 类型为 image/jpeg，并返回生成器

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False,threaded=True)
