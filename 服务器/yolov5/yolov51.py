# -------------------------------------#
#       cv2.waitKey() 函数的作用是等待用户按下一个键盘按键，并且它会暂停程序的执行，直到按键被按下。它接收一个参数，表示等待的时间（毫秒），如果设置为0或不设置，则会一直等待直到按键被按下。
#
# 因此，在你的代码中，cv2.imshow("send", frame) 用于显示图像窗口，然后通过 cv2.waitKey(1) 等待用户按下一个键盘按键，从而保持图像窗口的显示状态，直到用户按下键盘按键或超过1毫秒的时间。
#
# 这种方式允许你在每次循环迭代中显示图像并保持图像窗口的显示状态，而不会立即关闭窗口。这在实时图像显示和处理中非常有用，因为你可以看到图像的变化并进行交互。


#想把视频发送到哪里，就用目标端口的套接字 udpClientSock.sendto(image_data, (目标IP,目标端口))

# -------------------------------------#

from PIL import Image
import numpy as np
import cv2
import time
from socket import *
from time import ctime
from io import BytesIO
from PIL import Image
from predict import predict_


# 本进程的ip+端口号，本进程接收图片，检测后并将目标信息（类别，左上和右下两点坐标）发送给app
HOST = '0.0.0.0'  # 主机号为空白表示可以使用任何可用的地址。
PORT = 8081  # 端口号
BUFSIZ = 300000  # 接收数据缓冲大小
ADDR = (HOST, PORT)
udpSerSock = socket(AF_INET, SOCK_DGRAM)
udpSerSock.bind((HOST, PORT))
fps = 4
print("=====================Timestamp UDP server=====================")
#while True:
 #   print(1)
  #  request, client_addr = udpSerSock.recvfrom(4)
   # print(request)
    #if request == b"Request":  # 直接比较二进制数据
     #   print("Received request from {}:{}".format(client_addr[0], client_addr[1]))
        # 记录客户端信息
      #  break

#
# # 主程序ip+端口号
# HOST_send = '60.204.139.74'
# PORT_send = 8080  # 端口号
# UFSIZ_send = 4092  # 接收消息的缓冲大小
# ADDR_send = (HOST_send, PORT_send)
# udpCliSock1 = socket(AF_INET, SOCK_DGRAM)  # 创建客户端套接字
# remote
# udpCliSock_send.sendto(bytes('start_1', 'utf-8'), ADDR_send)
# flag = True
num = 0
skip = True
while True:
    num += 1
    if num % fps == 0:
        skip = False
    else:
        skip = True
    # print('Waiting to receive data...')
    #udpSerSock.settimeout(10)
    data_org, addr = udpSerSock.recvfrom(BUFSIZ)  # 连续接收指定字节的数据，接收到的是字节数组
    # print(data_org)
    # img = cv2.imdecode(data_org,cv2.IMREAD_COLOR)

    data = data_org.hex()
    #head_org = data_org[5:]

    # head = head_org.hex()

    # img=cv2.imdecode(head_org,cv2.IMREAD_COLOR)
    buf = BytesIO(data_org)
    # retval = cv2.imdecode(buf, 1)
    frame = np.array(Image.open(buf))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # print(img)
    if not skip:
        # if flag:
        #     udpCliSock_send.sendto(bytes('start_2', 'utf-8'), ADDR_send)
        #     flag = False
        # t1 = time.time()

        info = predict_(frame)

        #cv2.imshow('yolov5', frame)
        #cv2.waitKey(200)

        print(info)
        #
        # send_data = str(info)
        # udpCliSock1.sendto(send_data.encode("utf-8"), ADDR_send)
        # time.sleep(0.15)
    #else:
     #   cv2.imshow('yolov5', frame)
      #  cv2.waitKey(10)

    # 将图像数据发送到服务器8888端口，以待app获取
    encoded_image = cv2.imencode('.jpg', frame)[1]
    image_data = encoded_image.tobytes()
    udpClientSock.sendto(image_data, client_addr)
