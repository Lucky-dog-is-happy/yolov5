#! /usr/bin/env python3
# 导入模块
import argparse
import os
import platform
import sys
from pathlib import Path

import torch
# 确认文件路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) 
# 导入模块
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
# 检测模型版本
@smart_inference_mode()
# 传入参数
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    # 检测依赖
    check_requirements(exclude=('tensorboard', 'thop'))
    # 正式运行
    run(**vars(opt))
# 运行函数
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / '0',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
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
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    # 判断文件是否错误
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')
    # 保存路径
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # 加载模型
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    # 检测图片大小是否32k
    imgsz = check_img_size(imgsz, s=stride)
    # 批次大小
    bs = 1
    # 数据集
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    # 给gpu热热身
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    # 计数器、窗口信息、性能分析初始化
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        # 记录预处理时间
        with dt[0]:
            # 转换图像格式
            im = torch.from_numpy(im).to(model.device)
            # 数据类型转换
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            # 归一化
            im /= 255  # 0 - 255 to 0.0 - 1.0
            # 增加批次维度
            if len(im.shape) == 3:
                # 炫技是吧，欺负我没见识
                im = im[None]  # expand for batch dim
        with dt[1]:
            # 保证路径正确
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # 推理执行并可视化
            pred = model(im, augment=augment, visualize=visualize)
        with dt[2]:
            # 非极大值抑制，保留最优结果
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
        for i, det in enumerate(pred):  # per image
            # 计数器加一
            seen += 1
            # 处理图像
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
        # 转化为path对象
        p = Path(p)  # to Path
        # 设置保存路径
        save_path = str(save_dir / p.name)  # im.jpg
        # 设置标签文件路径
        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
        # 打印图像尺寸
        s += '%gx%g ' % im.shape[2:]  # print string
        # 归一化增益
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        # 处理裁剪图像
        imc = im0.copy() if save_crop else im0  # for save_crop
        # 绘制边框
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        # 检测非空
        if len(det):
                # Rescale boxes from img_size to im0 size
                # 缩放图像框
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, 5].unique():
                    # 计算检测数量
                    n = (det[:, 5] == c).sum()  # detections per class
                    # 格式化加入字符串
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                for *xyxy, conf, cls in reversed(det):
                    # 保存检测结果
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    # 绘制边框
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    # 保存裁剪图像
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                # 获得框后的图像
                im0 = annotator.result()
                # 显示图像
                cv2.imshow(str(p), im0)
                # 等待键盘输入
                cv2.waitKey(1)  # 1 milliseconds
            
                # 保存对象
                cv2.imwrite(save_path, im0)
                # 日志记录
                LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
    # 计算处理时间
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # 求速度总和
    LOGGER.info(f'速度: %.1fms 预处理, %.1fms 推理, %.1fms 非极大值抑制 每张图像的形状为 {(1, 3, *imgsz)}' % t)
    # 保存
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"保存地址 {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
# 主函数
if __name__=="__main__":
    opt=parse_opt()
    main(opt)
