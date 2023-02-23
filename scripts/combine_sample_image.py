"""
    Combine image in data folder to display in rows and cols
"""


import os
import random
from PIL import Image


def get_all_image(img_dir, shuffle = True):
    list_image_path = []
    for train_test_dir in os.listdir(img_dir):
        class_dir_path = os.path.join(img_dir, train_test_dir)
        for classname in os.listdir(class_dir_path):
            one_class_dir = os.path.join(class_dir_path, classname)
            for image in os.listdir(one_class_dir):
                image_path = os.path.join(one_class_dir, image)
                list_image_path.append(image_path)
    
    if shuffle == True:
        random.shuffle(list_image_path)
    return list_image_path


def pil_image_concat(list_img: list, rows: int, cols: int, image_size = (224, 224), save_name = "sample.jpg"):
    assert rows * cols == len(list_img)
    dst = Image.new('RGB', (image_size[0] * cols, image_size[1] * rows))
    for row in range(rows):
        for col in range(cols):
            img = Image.open(list_img[col + row * cols]).resize(image_size)
            dst.paste(img, (image_size[0] * col, image_size[1] * row))
    dst.save(save_name)


if __name__ == "__main__":
    rows, cols = 4, 7
    img_dir = "/home/teamai/TeamAI/dongtrinh/Fast-Food-Classification/data"
    list_image_path = get_all_image(img_dir)[:rows*cols]
    pil_image_concat(list_image_path, rows, cols)

