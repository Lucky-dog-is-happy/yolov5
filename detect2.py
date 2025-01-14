import os #与操作系统交互的模块
import sys #与Python解释器交互的模块
from pathlib import Path #处理文件路径的模块
import torch #pytorch模块

FILE = Path(__file__).resolve() #获取当前文件的绝对路径
ROOT = FILE.parents[0] #获取当前文件的父目录
if str(ROOT) not in sys.path: #如果当前文件的父目录不在模块查询路径中
    sys.path.append(str(ROOT)) #将当前文件的父目录添加到模块查询路径中
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative，绝对路径转换为相对路径
from utils.plots import Annotator, colors #标注边框、标签和生成颜色模块
from models.common import DetectMultiBackend #加载和推理多种模型模块
from utils.dataloaders import LoadStreams #加载视频流模块
# check_img_size检查图像尺寸是否为32的倍数\check_imshow检查是否显示图像\check_requirements检查是否满足要求/OpenCV/非极大值抑制，保留置信度最大边框/缩放框
from utils.general import (check_img_size,check_imshow,check_requirements,cv2,non_max_suppression,scale_boxes)
# select_device选择设备\smart_inference_mode智能推理模式\time_sync时间同步
from utils.torch_utils import select_device, smart_inference_mode,time_sync

class YOLOdetect:
    def __init__(self,source=0,weights=ROOT / "yolov5s.pt",data=ROOT / "data/YOLO_train.yaml"):
        self.source=source # 默认摄像头是0，如果要改成照片就输入照片路径作为参数 
        self.weights=weights # 模型路径
        self.data=data # 数据集路径
        self.device=select_device("0") # 默认使用GPU
        self.conf_thres=0.25 # 置信度，检测结果正确的概率
        self.iou_thres=0.45 # 交并比阈值，设置非极大值抑制的参数
        self.max_det=1000 # 最大框数
        self.line_thickness=3 # 框线粗细
        self.imgsz=(640, 640) # 推理图像尺寸（高度，宽度）

    def load(self):
        # 加载模型
        self.model = DetectMultiBackend(self.weights, device=self.device, data=self.data)
        # 模型步幅(32)/检测类型名称的列表/模型是否是pytorch
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        # 检测图片是不是步幅的倍数
        self.imgsz = check_img_size(self.imgsz, s=self.stride)
        # 加载视频流（视频流的路径/调整输入图像大小/指定步幅/自动调整为pytorch步幅/视频帧率步长）
        self.dataset = LoadStreams(str(self.source), img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=1)

    def preprocess(self,image0,image1):
        # 把图片从Numpy数组转化为torch张量并移到设备上
            image1 = torch.from_numpy(image1).to(self.device) 
            if self.model.fp16: # 如果使用FP16半精度推理
                image1 = image1.half()  # 张量转化为半精度
            else: # 如果不使用FP16半精度推理
                image1 = image1.float() # 张量转为单精度
            image /= 255.0 # 归一化
            # 如果图片是三维的，就增加一个维度（批次维度）
            if len(image1.shape) == 3: image1=image[None]
            return image1;

    def draw(self,det,image0,annotator):
        num=0 # 初始化框的数量
        for *xyxy, conf, cls in reversed(det):
            num+=1 # 框的数量加1
            c = int(cls) # 索引类别
            center_x=int((xyxy[0]+xyxy[2])/2)#中心点坐标
            center_y=int((xyxy[1]+xyxy[3])/2)
            cv2.circle(image0,(center_x,center_y),5,colors(c),7) # 画中心点
            # 在中心点附近显示坐标
            cv2.putText(image0,f"({center_x},{center_y})",(center_x+5,center_y+5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0))
            # 生成标签和置信度
            label = f"{self.names[c]} {conf:.2f}"
            # 画框和标签
            annotator.box_label(xyxy, label, color=colors(c, True))#给标注器赋值
            # 输出框的信息
            self.s+=f" No.{num} {self.names[int(c)]} " 
            self.s+=f"center:{(center_x,center_y)};"
    def postprocess(self,pred,image0,image1,t):
        for i, det in enumerate(pred): # 遍历推理结果            self.seen+=1
            image = image0[i].copy() # 复制图像以便画框
            self.s=f"{self.seen}: " # 输出已处理图像帧数
            self.s += "{:g}x{:g} ".format(*image1.shape[2:]) # 追加图像尺寸信息 
            annotator = Annotator(image, line_width=self.line_thickness,example=str(self.names)) # 绘制框和标签          
            if len(det): # 判断有无框，存储每一张图的所有框
                # 坐标映射回原图大小
                det[:,:4] = scale_boxes(image1.shape[2:], det[:,:4], image.shape).round()
                # 统计每个类别框数量
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    self.s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
            self.draw(det,image,annotator) # 画框和中心点
            image = annotator.result() # 获取标注结果
            time_pass=time_sync()-t # 计算推理时间
            cv2.putText(image, f"fps:{1/time_pass:.2f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) # 显示fps
            self.s+=f"time:{time_pass*1000:.2f}ms" # 输出推理时间
            self.s+=f"fps:{1/time_pass:.2f}" # 输出fps
            print(self.s) # 打印信息
            cv2.imshow('result', image)
            cv2.waitKey(1)

    def inference(self): # 推理
        # 初始化计数器、窗口信息、性能分析
        self.seen, self.windows= 0, []
        self.model.warmup(imgsz=(1,3,*self.imgsz)) # gpu预热
        for path, image1, image0, vid_cap, printInfo in self.dataset:
            t=time_sync() # 记录当前时间，便于后续计算推理时间
            image1 = self.preprocess(image0,image1) # 图像预处理
            # 模型推理
            pred = self.model(image1, augment=False,visualize=False) # 模型推理
            # 非极大值抑制过滤，留下好的框
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False, max_det=self.max_det)
            self.postprocess(pred,image0,image1,t) # 后处理
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
    @smart_inference_mode()
    def run(self):
        self.load() # 加载模型
        t0=time_sync() # 记录开始时间
        self.inference() # 推理
        print(f"Total time:{time_sync()-t0:.2f}s")

def main():
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))#检测有没有依赖包
    yolo=YOLOdetect(source=0)
    yolo.run()

if __name__ == '__main__':
    main()
         
