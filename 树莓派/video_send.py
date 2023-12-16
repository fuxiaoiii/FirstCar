from socket import *
import time
import cv2

time.sleep(5)
print("=====================发送数据时间戳UDP服务器=====================")

# 目标检测程序的ip+端口号
HOST = '116.63.176.132'  # 主机号为空白表示可以使用任何可用的地址。60.204.139.74
PORT = 8081  # 端口号
ADDR_send = (HOST, PORT)
udpCliSock_send = socket(AF_INET, SOCK_DGRAM)  # 创建客户端套接字

capture = cv2.VideoCapture(0)

while True:
    _, frame = capture.read()
    frame = cv2.resize(frame, (400, 320))
    # 发送图片给检测程序
    img_encode = cv2.imencode('.jpg', frame)[1]
    data = img_encode.tobytes()

    try:
        udpCliSock_send.sendto(data, ADDR_send)
    except OSError as e:
        if e.errno == 90:
            print("Message too long. Waiting and retrying...")
            #time.sleep(0.1)  # 等待一段时间后重试
        else:
            print(f"Error: {e}")
    
    time.sleep(0.25)

udpCliSock_send.close()
