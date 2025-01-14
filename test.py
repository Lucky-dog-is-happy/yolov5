import cv2

# 打开摄像头设备

cap = cv2.VideoCapture('/dev/video0')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 循环读取视频帧
while True:
    # 读取一帧
    ret, frame = cap.read()

    # 如果读取失败，退出循环
    if not ret:
        print("无法读取视频帧")
        break

    # 显示视频帧
    cv2.imshow('Camera', frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()