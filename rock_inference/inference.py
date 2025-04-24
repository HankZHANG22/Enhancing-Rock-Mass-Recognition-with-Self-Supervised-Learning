import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import faiss
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import config as cfg
from models import get_model
from utils.utils import cosine_sim, ResizeAndPad
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import cv2
from ultralytics import YOLO

# 加载微调后的 YOLOv8 模型
detection_model = YOLO(cfg.detection_version + ".pt")

model = get_model(cfg.backbone, num_features=cfg.embedding_size)
model.load_state_dict(torch.load(os.path.join("weights", cfg.model_version + ".pt")))
model = model.cuda().eval()
print("Loaded: backbone: {} model: {}".format(cfg.backbone, cfg.model_version))

# 特征提取函数
def inference(image):
    transform = transforms.Compose([
        ResizeAndPad((cfg.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = transform(image).unsqueeze(0).cuda()
    with torch.no_grad():
        feature = model(image)
    return feature.cpu().numpy()

# 多线程处理的辅助函数
def process_image(img_path, class_name):
    image = Image.open(img_path)
    feature = inference(image)
    return feature, img_path, class_name

# 使用多线程分批构建FAISS索引
def build_faiss_index(image_dir, num_threads=96, batch_size=2000):
    d = cfg.embedding_size  # 特征维度
    index = faiss.IndexFlatL2(d)  # 使用L2距离的索引
    image_paths = []
    labels = []

    futures = []
    batch_features = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for class_name in os.listdir(image_dir):
            class_path = os.path.join(image_dir, class_name)
            if os.path.isdir(class_path):  # 仅处理文件夹
                for file in os.listdir(class_path):
                    if file.endswith(('.jpg', '.png', '.jpeg', 'JPG', "JPEG", "PNG")):
                        img_path = os.path.join(class_path, file)
                        futures.append(executor.submit(process_image, img_path, class_name))
                    
                    # 一旦达到批次大小，处理结果并清空内存
                    if len(futures) >= batch_size:
                        for future in as_completed(futures):
                            feature, img_path, class_name = future.result()
                            batch_features.append(feature)
                            image_paths.append(img_path)
                            labels.append(class_name)
                            print(f"Processed image: {img_path}")
                        
                        # 将批次特征添加到FAISS索引
                        batch_features = np.vstack(batch_features)
                        index.add(batch_features)
                        print(f"Added batch to FAISS index, batch size: {len(batch_features)}")

                        # 清空列表以释放内存
                        futures = []
                        batch_features = []

        # 处理最后一批未满的任务
        if futures:
            for future in as_completed(futures):
                feature, img_path, class_name = future.result()
                batch_features.append(feature)
                image_paths.append(img_path)
                labels.append(class_name)
                print(f"Processed image: {img_path}")
            
            batch_features = np.vstack(batch_features)
            index.add(batch_features)
            print(f"Added final batch to FAISS index, batch size: {len(batch_features)}")

    return index, image_paths, labels

# 查询最相似的TOP5图片
def search_top_k(index, query_feature, k=5):
    D, I = index.search(query_feature, k)
    return I, D

# 构建FAISS索引并存储训练集特征
def generate_faiss_index(train_image_dir = "/home/langchao16/data/test_recognition/rock/"):
    index, image_paths, labels = build_faiss_index(train_image_dir)
    faiss.write_index(index, "rock_index.index")

    # 保存路径和标签
    np.save("image_paths.npy", image_paths)
    np.save("labels.npy", labels)

# 加载索引和元数据
def load_faiss_and_metadata(index_path, image_paths_path, labels_path):
    index = faiss.read_index(index_path)
    image_paths = np.load(image_paths_path, allow_pickle=True).tolist()
    labels = np.load(labels_path, allow_pickle=True).tolist()
    return index, image_paths, labels


# 查询新图片的特征并在FAISS数据库中检索TOP5最相似图片
def singal_image_test(query_image_path, rock_index_path, image_paths_path, labels_path):
    query_image = Image.open(query_image_path)
    query_feature = inference(query_image)

    index, image_paths, labels = load_faiss_and_metadata(rock_index_path, image_paths_path, labels_path)

    top_k_indices, distances = search_top_k(index, query_feature, k=5)
    
    for i in range(5):
        print(f"Image: {image_paths[top_k_indices[0][i]]}, Label: {labels[top_k_indices[0][i]]}, Distance: {distances[0][i]}")

# 查询最相似的TOP5类别
def process_and_search_image(image_path, index, labels, k=5):
    query_image = Image.open(image_path)
    query_feature = inference(query_image)
    top_k_indices, _ = search_top_k(index, query_feature, k)

    top_k_labels = []
    for i in range(k):
        top_k_labels.append(labels[top_k_indices[0][i]])

    return top_k_labels

# 换行在图片上打印文字
def draw_text_with_wrapping(draw, text, position, font, max_width, color=(255, 0, 0)):

    lines = []
    words = text.split('，')  # 分割文本，以逗号为分隔符
   
    while words:
        line = words.pop(0)
        lines.append(line.strip())  # 移除行尾的空格

    # 在图像上逐行绘制文本
    x, y = position
    for line in lines:
        draw.text((x, y), line, font=font, fill=color)
        y += font.getbbox(line)[3] - font.getbbox(line)[1]  # 获取文本的高度并移动到下一行的位置

    # lines = []
    # words = text.split('，')
    # while words:
    #     line = ''
    #     while words and font.getbbox(line + words[0])[2] <= max_width:
    #         line += (words.pop(0) + ' ')
    #     if not line:  # 如果行为空，说明当前单词太长，直接添加到行内然后截断
    #         line = words.pop(0)
    #     lines.append(line.strip())

    # # 在图像上逐行绘制文本
    # x, y = position
    # for line in lines:
    #     draw.text((x, y), line, font=font, fill=color)
    #     y += font.getbbox(line)[3] - font.getbbox(line)[1]  # 获取文本的高度并移动到下一行的位置

# 调整图片大小
def resize_image(image, target_size):
    return image.resize(target_size, Image.Resampling.LANCZOS)

def clear_directory(directory_path):
    """
    清空指定目录下的所有文件，但保留目录结构。

    :param directory_path: 需要清空的目录路径
    """
    if not os.path.exists(directory_path):
        return
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"删除文件 '{file_path}' 时出错: {e}")

# 拼接图片
def combine_image(image_out_dir, subdir, file, query_image, class_name, top_k_indices, labels, images):
    top_k_labels = [labels[i] for i in top_k_indices[0][:3]]
    top_k_images = [Image.open(images[i]) for i in top_k_indices[0][:3]]

    # 找出最大尺寸
    max_width = max([query_image.width] + [img.width for img in top_k_images])
    max_height = max([query_image.height] + [img.height for img in top_k_images])
    target_size = (max_width, max_height)
                            
    # 调整所有图片到最大尺寸
    query_image = resize_image(query_image, target_size)
    top_k_images = [resize_image(img, target_size) for img in top_k_images]

    # 拼接原图和Top 1、Top 2、Top 3图片
    total_width = query_image.width + sum(img.width for img in top_k_images)
    max_height = max(query_image.height, *[img.height for img in top_k_images]) + 1500
    combined_image = Image.new('RGB', (total_width, max_height))
    combined_image.paste(query_image, (0, 0))

    x_offset = query_image.width
    for img in top_k_images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width
                            
    # 在图片下方添加文字说明
    draw = ImageDraw.Draw(combined_image)
    # 指定字体文件和大小
    font = ImageFont.truetype("font/STSong.ttf", 150)
    text = f"图片文件名：{file}，原类别：{class_name}，识别出Top1类别：{top_k_labels[0]}，识别出Top2类别：{top_k_labels[1]}，识别出Top3类别：{top_k_labels[2]}"
    text_start_position = (10, max_height - 1300)  # Adjust the position as needed
    draw_text_with_wrapping(draw, text, text_start_position, font, total_width, color=(255, 255, 255))
                           
    # 保存拼接后的图片
    combined_image.save(os.path.join(image_out_dir, f"{subdir}_{class_name}_{file}"))
    print(f"Saved combined image")


# 查询指定路径文件夹下的所有图片
def singal_dir_test(query_dir,  excel_output_dir, image_out_dir, rock_index_path="rock_index.index", image_paths_path="image_paths.npy", labels_path="labels.npy"):
    index, images, labels = load_faiss_and_metadata(rock_index_path, image_paths_path, labels_path)
    for subdir in os.listdir(query_dir):
        subdir_path = os.path.join(query_dir, subdir)
        if os.path.isdir(subdir_path):
            excel_data = []
            for class_name in os.listdir(subdir_path):
                class_path = os.path.join(subdir_path, class_name)
                if os.path.isdir(class_path):
                    for file in os.listdir(class_path):
                        if file.endswith(('.jpg', '.png', '.jpeg', 'JPG', 'JPEG', 'PNG')):
                            img_path = os.path.join(class_path, file)
                            query_image = Image.open(img_path)

                            # 截取出岩石部分再分类
                            results = detection_model.predict(img_path)
                            # 获取检测结果
                            for idx, result in enumerate(results[0].boxes):
                                # 提取检测的边界框（Bounding Box）信息
                                x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())

                                # 读取原图
                                img = cv2.imread(img_path)

                                # 裁剪检测到的岩石区域
                                cropped_img = img[y1:y2, x1:x2]
                                # 在裁剪后的图像部分，添加这一行，将 numpy 数组转换为 PIL.Image
                                cropped_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                                
                                try:
                                    query_feature = inference(cropped_img)
                                    top_k_indices, _ = search_top_k(index, query_feature, k=5)
                            
                                    top_k_labels = [labels[i] for i in top_k_indices[0]]
                                    excel_data.append({
                                        "文件名": file,
                                        "文件路径": img_path,
                                        "原类别": class_name,
                                        "分类Top 1": top_k_labels[0],
                                        # "对应图片路径": images[0],
                                        "分类Top 2": top_k_labels[1],
                                        # "Path 2": images[1],
                                        "分类Top 3": top_k_labels[2],
                                        # "Path 3": images[2],
                                        "分类Top 4": top_k_labels[3],
                                        # "Path 4": images[3],
                                        "分类Top 5": top_k_labels[4],
                                        # "Path 5": images[4],
                                    })
                                    combine_image(image_out_dir, subdir, file, query_image, class_name, top_k_indices, labels, images)
                                except RuntimeError as e:
                                    print(img_path)
                                    print(f"An error occurred: {e}")
                                except OSError as e:
                                    print(img_path)
                                    print(f"An error occurred: {e}")
                            
            # 创建DataFrame并保存到Excel文件
            df = pd.DataFrame(excel_data)
            output_file = os.path.join(excel_output_dir, f"{subdir}.xlsx")
            df.to_excel(output_file, index=False)
            print(f"Saved results to {output_file}")


if __name__ == '__main__':
    # python inference.py --build_faiss False --faiss_dir /home/langchao16/data/province/fujian/vali --query_dir /home/langchao16/data/准泛化测试样本2024 --excel_output_dir /home/langchao16/code/rock_inference/实验结果 --image_out_dir /home/langchao16/code/rock_inference/拼接图片

    parser = argparse.ArgumentParser()
    parser.add_argument('--build_faiss', type=bool, default=False, required=True, help='Specify wether build faiss.')
    parser.add_argument('--faiss_dir', required=True, help='Specify the query directory.')
    parser.add_argument('--query_dir', required=True, help='Specify the query directory.')
    parser.add_argument('--excel_output_dir', required=True, help='Specify the output directory.')
    parser.add_argument('--image_out_dir', required=True, help='Specify the output directory.')
    args = parser.parse_args()

    # 指定的查询文件夹和输出目录
    build_faiss = args.build_faiss
    faiss_dir = args.faiss_dir
    query_dir = args.query_dir
    excel_output_dir = args.excel_output_dir
    image_out_dir = args.image_out_dir

    if build_faiss:
        print("执行faiss数据库构建")
        generate_faiss_index(faiss_dir)
    else:
        # 如果 build_faiss 为 False，不执行某项操作
        print("不执行faiss数据库构建，因为 build_faiss 为 False")

    # 确保输出目录存在
    os.makedirs(excel_output_dir, exist_ok=True)
    os.makedirs(image_out_dir, exist_ok=True)
    clear_directory(image_out_dir)
    clear_directory(excel_output_dir)

    singal_dir_test(query_dir, excel_output_dir, image_out_dir)