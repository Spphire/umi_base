import numpy as np
import cv2

def preview_noise(image_path):
    # 读取一张你的 Wrist 图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img.astype(np.float32) / 255.0

    scales = [0.05, 0.1, 0.2, 0.5]
    results = []

    for s in scales:
        noise = np.random.normal(0, s, img_float.shape)
        noisy_img = np.clip(img_float + noise, 0, 1)
        # 转换回 uint8 方便显示
        noisy_img = (noisy_img * 255).astype(np.uint8)
        results.append(cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR))
        cv2.putText(results[-1], f"sigma={s}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 横向拼接结果
    combined = np.hstack(results)
    cv2.imshow("Noise Sensitivity Test", combined)
    cv2.waitKey(0)

preview_noise('.cache/left_wrist_img_0.png')