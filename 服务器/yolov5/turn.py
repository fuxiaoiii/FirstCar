import socket
import time
# 本地绑定信息
local_ip = '0.0.0.0'
local_port = 8888

# 创建UDP套接字
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.bind((local_ip, local_port))

print("UDP socket listening on {}:{}".format(local_ip, local_port))

while True:
    request, client_addr = udp_socket.recvfrom(1024)
    if request == b"Request":  # 直接比较二进制数据
        print("Received request from {}:{}".format(client_addr[0], client_addr[1]))
        # 记录客户端信息
        data = b"Connect Successfully"
        udp_socket.sendto(data, client_addr)
        break

while True:
    data, _ = udp_socket.recvfrom(1024*1024)  # 最大接收数据大小
    # # 转发视频流给客户端
    # print(1)
    # udp_socket.sendto(data, (client_ip, client_port))
    #print(data)
    #data = b"ho"
    udp_socket.sendto(data, client_addr)
    time.sleep(0.1)

# 关闭套接字
udp_socket.close()
