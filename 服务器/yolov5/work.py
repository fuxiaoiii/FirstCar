from flask import Flask, Response
import cv2
import requests
import numpy as np


app = Flask(__name__)

# 模拟视频流处理函数
def process_frame(frame):
    return frame

# 用于实时传输处理后视频流的路由
@app.route('/processed_video_feed')
def processed_video_feed():
    stream_url = 'http://119.3.225.185:9000/?action=stream'

    def generate():
        while True:
            # 从网络流地址读取视频帧数据
            response = requests.get(stream_url, stream=True)
            bytes_data = bytes()
            for chunk in response.iter_content(chunk_size=8192):
                bytes_data += chunk

                # 解码视频帧
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]

                    # 将视频帧转换为 NumPy 数组
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                    # 对视频帧进行处理
                    processed_frame = process_frame(frame)

                    # 将处理后的视频帧转换为 JPEG 格式
                    ret, jpeg = cv2.imencode('.jpg', processed_frame)

                    # 将处理后的视频帧作为 Response 内容返回给客户端
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    # 设置 Response 的 MIME 类型为 image/jpeg
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 启动 Flask 应用，监听在服务器的 0.0.0.0:5000 端口上
    app.run(host='0.0.0.0', port=5000, debug=False)
