import socket
local_ip = '0.0.0.0'
local_port = 8888


udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.bind((local_ip, local_port))

print("Listening on port {}:{}".format(local_ip, local_port))

while True:
    try:
        data, _ = udp_socket.recvfrom(65507)  
        print("Received data. Size: {} bytes".format(len(data)))
    except KeyboardInterrupt:
        break

udp_socket.close()
