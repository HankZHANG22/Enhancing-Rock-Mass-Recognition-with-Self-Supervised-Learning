import os
from PIL import Image


def delete_corrupted_images(root_directory):
    # 遍历根目录及其子目录下的所有文件
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            # 构造完整的文件路径
            file_path = os.path.join(dirpath, filename)

            try:
                # 尝试打开图像文件
                with Image.open(file_path) as img:
                    # 确保图像被成功打开和验证
                    img.verify()  # 这将检查文件的合法性
            except (IOError, ValueError) as e:
                # 如果发生IOError或ValueError，说明文件无法打开或读取
                print(f"Corrupted image found: {filename}")
                # 删除损坏的图像文件
                os.remove(file_path)


if __name__ == "__main__":
    delete_corrupted_images("/data/app/image_folder")
