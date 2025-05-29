import requests
import cv2
import numpy as np

def main():
    url = 'http://localhost:1280/video-feed'  # 根据服务器地址调整
    boundary = '--frame'
    try:
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            print(f"无法连接到服务器，状态码：{response.status_code}")
            return
        
        bytes_buffer = b''
        frame_count = 0
        rgb_image = None  # 用于存储当前帧的 RGB 图像

        for chunk in response.iter_content(chunk_size=1024):
            if not chunk:
                continue
            bytes_buffer += chunk

            while True:
                # 查找边界
                boundary_index = bytes_buffer.find(boundary.encode())
                if boundary_index == -1:
                    break
                # 查找下一个部分的边界
                next_boundary = bytes_buffer.find(boundary.encode(), boundary_index + len(boundary))
                if next_boundary == -1:
                    break
                # 提取当前部分
                part = bytes_buffer[boundary_index:next_boundary]
                bytes_buffer = bytes_buffer[next_boundary:]

                # 解析头部
                header_end = part.find(b'\r\n\r\n')
                if header_end == -1:
                    continue
                headers = part[:header_end].decode('utf-8', errors='ignore').split('\r\n')
                frame_count = None
                jpeg_data = None
                for header in headers:
                    if header.startswith('X-Frame-Count:'):
                        try:
                            frame_count = int(header.split(':')[1].strip())
                        except ValueError:
                            frame_count = None
                        break
                if frame_count is not None:
                    # 提取 JPEG 数据
                    jpeg_start = header_end + 4
                    jpeg_end = part.find(b'\r\n', jpeg_start)
                    jpeg_data = part[jpeg_start:].rstrip(b'\r\n')
                    if jpeg_data:
                        # 解码 JPEG 数据为图像
                        img_array = np.frombuffer(jpeg_data, dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if img is not None:
                            # 转换为 RGB
                            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            # 输出帧计数和图像信息
                            print(f"接收到帧计数：{frame_count}，图像尺寸：{rgb_image.shape}, 通道数：{rgb_image.shape[2]}")
                            # # 这里可以对 rgb_image 进行进一步处理，如保存、分析等
                            # cv2.imwrite(f'frame_{frame_count}.jpg', rgb_image)
                            # exit()
        print("连接关闭")
    except Exception as e:
        print(f"发生错误：{e}")

if __name__ == '__main__':
    main()