import os
import cv2
from ultralytics import YOLO

from config import config as cfg

# 加载微调后的 YOLOv8 模型
model = YOLO(cfg.detection_version + ".pt")

# 输入图片的根目录
input_root_dir = '/home/langchao16/data/train_set_resized_origin'

# 输出保存裁剪图像的根目录
output_root_dir = '/home/langchao16/data/train_set_resized_origin_detected'

# 遍历所有子文件夹和图片
for subdir, dirs, files in os.walk(input_root_dir):
    for file in files:
        # 仅处理图片文件（可以根据需要扩展图片格式）
        if file.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
            input_file_path = os.path.join(subdir, file)
            print(f"Processing {input_file_path}")

            # 使用模型检测图片中的目标
            results = model.predict(input_file_path)

            # 获取检测结果
            for idx, result in enumerate(results[0].boxes):
                # 提取检测的边界框（Bounding Box）信息
                x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())

                # 读取原图
                img = cv2.imread(input_file_path)

                # 裁剪检测到的岩石区域
                cropped_img = img[y1:y2, x1:x2]

                # 获取原文件的类别子文件夹
                relative_path = os.path.relpath(subdir, input_root_dir)
                output_dir = os.path.join(output_root_dir, relative_path)

                # 确保输出目录存在
                os.makedirs(output_dir, exist_ok=True)

                # 保存裁剪后的图像，命名方式为：原文件名_索引.扩展名
                file_name, file_ext = os.path.splitext(file)
                output_file_name = f"{file_name}_{idx}{file_ext}"
                output_file_path = os.path.join(output_dir, output_file_name)

                # 保存裁剪的图像
                cv2.imwrite(output_file_path, cropped_img)

print("All images processed successfully.")
