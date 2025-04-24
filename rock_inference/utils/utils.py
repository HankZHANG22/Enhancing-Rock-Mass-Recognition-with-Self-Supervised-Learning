import cv2
import numpy as np
from PIL import Image


def resize_and_pad(img, input_size=448):
    h, w, _ = img.shape
    if h == w:
        img = cv2.resize(img, (input_size, input_size))
        return img
    if h > w:
        new_h = input_size
        new_w = int(w / h * input_size)
    else:
        new_w = input_size
        new_h = int(h / w * input_size)
    img = cv2.resize(img, (new_w, new_h))
    # 在短边两侧填充0，使得图像变成正方形
    if h > w:
        img = np.pad(img, ((0, 0), ((input_size - new_w) // 2, input_size - new_w - (input_size - new_w) // 2), (0, 0)), 'constant', constant_values=0)
    else:
        img = np.pad(img, (((input_size - new_h) // 2, input_size - new_h - (input_size - new_h) // 2), (0, 0), (0, 0)), 'constant', constant_values=0)
    return img


def cosine_sim(feature1, feature2):
    return np.inner(feature1, feature2) / np.dot(np.linalg.norm(feature1, axis=1, keepdims=True),
                                                np.linalg.norm(feature2, axis=1, keepdims=True).T)

class ResizeAndPad:
    # Resize the image to the target size while keeping the aspect ratio, then pad the resized image to the target size with black pixels
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        # Calculate the aspect ratio
        width, height = img.size
        aspect_ratio = width / height

        # Calculate the new size with the aspect ratio preserved
        if aspect_ratio > 1:
            new_width = self.target_size
            new_height = int(self.target_size / aspect_ratio)
        elif aspect_ratio < 1:
            new_width = int(self.target_size * aspect_ratio)
            new_height = self.target_size
        else:
            return img.resize((self.target_size, self.target_size))

        # Resize the image while preserving the aspect ratio
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        # Create a new blank image of the target size and paste the resized image in the center
        padded_img = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))
        padded_img.paste(resized_img, ((self.target_size - new_width) // 2, (self.target_size - new_height) // 2))

        return padded_img

