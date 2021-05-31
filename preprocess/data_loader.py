import pandas as pd
import json
import cv2
import os
import numpy as np

if __name__ == "__main__":
    file = "G:/archive/results.csv"
    fileOut = "F:/SGRAF/data/img.json"
    img_file_out = "../SGRAF/f30k_precomp/train_caps.txt"
    img_path_jpg = "G:/archive/flickr30k_images/"
    img_path_npy = "F:/SGRAF/image_npy/"
    df = pd.read_csv(file, sep="|")
    col = ["image_name", "comment_number", "comment"]
    img = dict()
    stack = list()
    for row in df.values:
        image_name, _, comment = row
        # 有些数据不规范，后面没有断句
        if not comment.endswith("."):
            comment += "."
        img.setdefault(image_name, list())
        img[image_name].append(comment.strip())
    with open(fileOut, 'w') as f:
        json.dump(img, f, ensure_ascii=False, indent=4)
    img_folder = os.listdir(img_path_jpg)
    with open(img_file_out, "w") as f:
        for img_file_name in img_folder:
            if img_file_name in img:
                f.write("".join(img[img_file_name]))
                f.write("\n")
                im1 = cv2.imread(img_path_jpg + img_file_name)
                np.save(img_path_npy + img_file_name.replace(".jpg", ".npy"), im1)
            else:
                print("file {image} not found".format(image=img_file_name))
