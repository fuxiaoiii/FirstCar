# -*- coding: utf-8 -*-
import paho.mqtt.client as mqtt
import RPi.GPIO as GPIO
import time
import serial
import json
import threading

current_mode = "control"

GPIO.setmode(GPIO.BCM)
global flag
flag = 0

# flag_lock = threading.Lock()




def led_control(temp):
    # 设置GPIO模式为BCM
    # GPIO.setmode(GPIO.BCM)

    # 将GPIO5和GPIO6设置为输出模式
    GPIO.setup(5, GPIO.OUT)
    GPIO.setup(6, GPIO.OUT)
    if temp == "on":
        # 将GPIO5和GPIO6的输出状态设置为高电平
        GPIO.output(5, GPIO.HIGH)
        GPIO.output(6, GPIO.HIGH)
    elif temp == "off":
        # 将GPIO5和GPIO6的输出状态设置为高电平
        GPIO.output(5, GPIO.LOW)
        GPIO.output(6, GPIO.LOW)


def getdistance():
    # 发出触发信号
    GPIO.output(17, GPIO.HIGH)
    # 保持10us以上
    time.sleep(0.000015)
    GPIO.output(17, GPIO.LOW)
    while not GPIO.input(18):
        pass
    # 发现高电平时开时计时
    t1 = time.time()
    while GPIO.input(18):
        pass
    # 高电平结束停止计时
    t2 = time.time()
    # 返回距离，单位为米
    return round((t2 - t1) * 340 / 2, 1)





def send_order(data):
    ser = serial.Serial('/dev/ttyAMA0', 9600, timeout=1)  # 根据实际情况修改串口名称和波特率

    # 配置串口
    print(ser.name)
    print(ser.port)
    print(ser.baudrate)
    print(ser.bytesize)
    print(ser.parity)  # 校验位N－无校验，E－偶校验，O－奇校验
    print(ser.stopbits)  # 停止位
    print(ser.timeout)  # 读超时设置
    print(ser.writeTimeout)  # 写超时
    print(ser.xonxoff)  # 软件流控
    print(ser.rtscts)  # 硬件流控
    print(ser.dsrdtr)  # 硬件流控
    print(ser.interCharTimeout)  # 字符间隔超时

    data_to_send = data

    # 发送数据
    ser.write(data_to_send.encode('utf-8'))
    time.sleep(0.5)

    # 接收并打印数据
    # while True:
    # if ser.in_waiting > 0:
    # received_data = ser.read_all().decode('utf-8').rstrip("\r\n")  # 解码并移除换行符
    # print(received_data)
    # break

    ser.flushInput()

    # 关闭串口连接
    ser.close()


# 设备身份信息
device_id = "64acf02bff79602237093ae3_rasp_0_0_2023071413"
username = "64acf02bff79602237093ae3_rasp"
password = "58ba9ddf98ef6c6095c4b4e8441049007e64d0b2ab415ef7ae868a1d9d7d0e71"

# MQTT服务器地址
mqtt_broker = "354ca305ab.iot-mqtts.cn-north-4.myhuaweicloud.com"
mqtt_port = 1883


# 连接回调函数
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker")
    else:
        print("Failed to connect, return code: ", rc)


def on_message(client, userdata, msg):
    global flag
    print("Received message: ", msg.payload.decode())
    global current_mode
    json_code = msg.payload.decode()

    # 解析JSON代码
    data = json.loads(json_code)

    # 判断json内容
    if "mode" in data["paras"]:
        current_mode = data["paras"]["mode"]
        print(current_mode)
    if current_mode == "control":
        flag = 0
        if "direction" in data["paras"]:
            direction = data["paras"]["direction"]
            # 进行相应的方向判断和处理
            print(direction)
            if direction == "forward":
                send_order("u")
            elif direction == "backward":
                send_order("d")
            elif direction == "turnleft":
                send_order("5")
            elif direction == "turnright":
                send_order("6")
            elif direction == "left":
                send_order("7")
            elif direction == "right":
                send_order("8")
            elif direction == "stop":
                send_order("s")
        elif "speed" in data["paras"]:
            speed = data["paras"]["speed"]
            if speed == "300":
                send_order("A")
            elif speed == "600":
                send_order("B")
            elif speed == "1000":
                send_order("C")
            elif speed == "1600":
                send_order("D")
        elif "switch" in data["paras"]:
            temp = data["paras"]["switch"]
            led_control(temp)
    elif current_mode == "automatic":
        # try:
        # GPIO.cleanup()
        # GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        # 第11号针，GPIO17
        GPIO.setup(17, GPIO.OUT, initial=GPIO.LOW)
        # 第12号针，GPIO18
        GPIO.setup(18, GPIO.IN)
        time.sleep(0.5)  # 初始化时间
        # 在设置 flag 之前获取锁
        # flag_lock.acquire()
        flag = 1
        # 在设置 flag 之后释放锁
        # flag_lock.release()

        print(flag)
    # except KeyboardInterrupt:
    # GPIO.cleanup()


# 创建MQTT客户端
client = mqtt.Client(client_id=device_id, clean_session=True)

# 设置连接和消息回调函数
client.on_connect = on_connect

client.on_message = on_message

# 设置用户名和密码
client.username_pw_set(username=username, password=password)




def loop():

    global flag
    print(flag)
    while True:
        # flag_lock.acquire()

        #time.sleep(0.2)
        if(flag==0):
            time.sleep(0.3)
            continue
        # if flag == 0:
        #     flag_lock.release()
        #     break
        # flag_lock.release()
        print(3)
        if getdistance() <= 1.0:
            if getdistance() < 0.5:
                send_order("u")
            elif getdistance() >= 0.5:
                send_order("s")
        elif getdistance() > 1.0:
            if getdistance() > 1.5:
                send_order("d")
            elif getdistance() <= 1.5:
                send_order("s")
        print("Distance:" + str(getdistance()) + "m")
        time.sleep(0.1)
    # if current_mode == "control":
    #    break

#loop线程
t = threading.Thread(target=loop)
t.start()

# 连接到MQTT服务器
client.connect(mqtt_broker, mqtt_port, keepalive=120)

# 订阅主题
topic = "command"
client.subscribe(topic)


# 循环处理MQTT网络流量
client.loop_forever()


t.join()
