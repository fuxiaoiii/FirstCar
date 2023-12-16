# -------------------------------------#
#       cv2.waitKey() 函数的作用是等待用户按下一个键盘按键，并且它会暂停程序的执行，直到按键被按下。它接收一个参数，表示等待的时间（毫秒），如果设置为0或不设置，则会一直等待直到按键被按下。
#
# 因此，在你的代码中，cv2.imshow("send", frame) 用于显示图像窗口，然后通过 cv2.waitKey(1) 等待用户按下一个键盘按键，从而保持图像窗口的显示状态，直到用户按下键盘按键或超过1毫秒的时间。
#
# 这种方式允许你在每次循环迭代中显示图像并保持图像窗口的显示状态，而不会立即关闭窗口。这在实时图像显示和处理中非常有用，因为你可以看到图像的变化并进行交互。


# 想把视频发送到哪里，就用目标端口的套接字 udpClientSock.sendto(image_data, (目标IP,目标端口))

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
BUFSIZ = 30000*4  # 接收数据缓冲大小
ADDR = (HOST, PORT)
udpSerSock = socket(AF_INET, SOCK_DGRAM)
udpSerSock.bind((HOST, PORT))
fps = 3
print("=====================Timestamp UDP server=====================")
while True:
    request, client_addr = udpSerSock.recvfrom(1024)
    # print(request)
    if request == b"Request":  # 直接比较二进制数据
        print("Received request from {}:{}".format(client_addr[0], client_addr[1]))
        # 记录客户端信息
        break

num = 0
skip = True

while True:
    num += 1
    if num % fps == 0:
        skip = False
    else:
        skip = True
    data, addr = udpSerSock.recvfrom(BUFSIZ)

    if data == b"Request":
        client_addr = addr
        print("Received request from {}:{}".format(client_addr[0], client_addr[1]))

    if not skip:
        buf = BytesIO(data)
        try:
            img = Image.open(buf)
            if img.format == 'JPEG':
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                info = predict_(frame)
                encoded_image = cv2.imencode('.jpg', frame)[1]
                data = encoded_image.tobytes()

                # 检查数据长度是否超过UDP的最大限制
                max_packet_size = 65507  # UDP数据包的最大大小
                if len(data) <= max_packet_size:
                    try:
                        udpSerSock.sendto(data, client_addr)
                    except OSError as e:
                        if e.errno == 90:
                            print("Message too long. Waiting 0.1s...")
                            time.sleep(0.1)
                        else:
                            print(f"OSError: {e}")
                else:
                    print("Image data too large to send in a single packet. Splitting not implemented.")

            else:
                print("Received image is not in JPEG format.")
        except Exception as e:
            print(f"Error processing image data: {e}")


    # 将图像数据发送到app
    print(len(data))
    udpSerSock.sendto(data, client_addr)
udpSerSock.close()
