# Copilot是神，救我狗命！！！ 

## 模型数据集要多，train和val各15份左右，epochs给他调1000次，暴力！！！ 
默认的100次练了和没练一样，confusion-matric给我一条横线是什么意思？模型失败的意思，因此我们要调大epochs参数和增加训练集个数

## plot的.getsize()要换成.getbbox()有四个参数，0-2，1-3就可以得到长和高
这个在运行train.py时要注意修改
### 运行语句：
0. conda activate yolov5 
运行**conda**环境
1.  python detect1.py --source 0
用摄像头检测，默认模型yolov5s.pt
2.  python train.py
首先我们要用[make sense](https://www.makesense.ai/)网站一个个手动标注数据集（妈的，真tm累），然后我们要把数据集按照以下树状图排列
```tree
yolov5-7.0/
├── datasets/
│   └── card/
│       ├── images/
│       │   ├── train/
│       │   │   ├── image1.jpg
│       │   │   ├── image2.jpg
│       │   │   └── ...
│       │   ├── val/
│       │   │   ├── image1.jpg
│       │   │   ├── image2.jpg
│       │   │   └── ...
│       └── labels/
│           ├── train/
│           │   ├── image1.txt
│           │   ├── image2.txt
│           │   └── ...
│           ├── val/
│           │   ├── image1.txt
│           │   ├── image2.txt
│           │   └── ...
├── data/
├── models/
├── utils/
├── detect.py
├── train.py
├── requirements.txt
└── README.md
```
然后写一个.yaml文件标注数据集的地址和标签类别，然后在train.py文件里手动修改data的默认参数为card.yaml，然后就在终端运行以上代码
3.  python detect.py --weights runs/train/exp2/weights/best.pt --source data/images/11.jpg --device 0
这个是用自己训练的模型去检测图片，（快把你的银行卡拿来试试[doge]）
--device 0说明用gpu检测

## 简化detect.py
首先看[b站视频](https://www.bilibili.com/video/BV1Dt4y1x7Fz/?spm_id_from=333.337.search-card.all.click)学习大概代码含义，接着看[知乎文章](https://zhuanlan.zhihu.com/p/501798155)学习大致如何用，如果你还不会？**Copilot**救你大命！！！
先照着看过几遍代码的感觉去简化，把什么网站，视频啥的尽量删掉，删掉报错就问AI，AI还是改错就不改了[doge]
然后运行一看，都是报错，照着报错用着AI去一次次修改，修改个区区3、4个小时大概就好了。
至于中文注释，问AI，AI解释不够生动，自己再做补充，就这样了
最后重申：

## Copilot救我狗命！！！