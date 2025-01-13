# 爱死Copilot了！！！
import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.plots import Annotator, colors#导入必要的yolov5模块
from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import (LOGGER, check_img_size, Profile, check_imshow,check_requirements,cv2,non_max_suppression,scale_boxes)
from utils.torch_utils import select_device, smart_inference_mode,time_sync

@smart_inference_mode()
def run(weights=ROOT / 'yolov5s.pt',  # 模型路径
        source='1',  # 摄像头
        data=ROOT / 'data/coco128.yaml',  # 数据集配置文件路径
        imgsz=(640, 640),  # 推理图像尺寸（高度，宽度）
        conf_thres=0.25,  # 置信度阈值
        iou_thres=0.45,  # NMS IOU阈值
        max_det=1000,  # 每张图像的最大检测数量
        device='0',  # CUDA设备，例如 0 或 0,1,2,3 或 cpu
        view_img=True,  # 显示结果
        line_thickness=3,  # 边界框厚度（像素）
        hide_labels=False,  # 隐藏标签
        hide_conf=False,  # 隐藏置信度
        half=False,  # 使用FP16半精度推理
        dnn=False,  # 使用OpenCV DNN进行ONNX推理
        vid_stride=1,  # 视频帧率步长
):
    # 加载模型
    device = select_device(device)  # 选择设备
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)  # 加载模型
    stride, names, pt = model.stride, model.names, model.pt  # 获取模型步长、类别名称和模型类型
    imgsz = check_img_size(imgsz, s=stride)  # 检查图像尺寸是否为32的倍数

    # 加载摄像头流
    try:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)  # 加载摄像头流
    except AssertionError as e:
        LOGGER.error(f"Failed to open camera: {e}")
        return

    bs = len(dataset)  # 获取批次大小
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # 给GPU热身

    # 计数器、窗口信息、性能分析初始化
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())  # 初始化计数器、窗口信息和性能分析

    for path, im, im0s, vid_cap, s in dataset:  # 遍历数据集
        # 记录预处理时间
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)  # 转换图像格式
            im = im.half() if model.fp16 else im.float()  # 数据类型转换
            im /= 255  # 归一化
            if len(im.shape) == 3:
                im = im[None]  # 增加批次维度

        # 推理
        with dt[1]:
            pred = model(im)  # 推理执行

        # 非极大值抑制，保留最优结果
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)

        for i, det in enumerate(pred):  # 遍历每张图像的检测结果
            seen += 1  # 计数器加一
            p, im0, frame = path[i], im0s[i].copy(), dataset.count  # 获取图像路径、原始图像和帧数
            p = Path(p)  # 转化为Path对象
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化增益
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))  # 绘制边框

            if len(det):  # 检测非空
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # 缩放图像框
                for *xyxy, conf, cls in reversed(det):  # 遍历检测结果
                    c = int(cls)  # 获取类别
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')  # 设置标签
                    annotator.box_label(xyxy, label, color=colors(c, True))  # 绘制边框和标签

            im0 = annotator.result()  # 获得框后的图像
            cv2.imshow(str(p), im0)  # 显示图像
            if cv2.waitKey(1) == ord('q'):  # 等待键盘输入，按下'q'键退出
                break

# 传入参数
def parse_opt():
    parser = argparse.ArgumentParser()  # 创建ArgumentParser对象
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')  # 添加weights参数
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob/screen/0(webcam)')  # 添加source参数
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')  # 添加data参数
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')  # 添加imgsz参数
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')  # 添加conf-thres参数
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')  # 添加iou-thres参数
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')  # 添加max-det参数
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')  # 添加device参数
    parser.add_argument('--view-img', action='store_true', help='show results')  # 添加view-img参数
    opt = parser.parse_args()  # 解析命令行参数
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # 扩展图像尺寸
    return opt  # 返回解析后的参数对象

# 主函数
def main(opt):
    run(**vars(opt))  # 调用run函数

if __name__ == "__main__":  # 如果是主程序
    opt = parse_opt()  # 解析命令行参数
    main(opt)  # 调用主函数