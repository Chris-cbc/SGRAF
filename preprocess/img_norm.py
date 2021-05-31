import cv2
import numpy as np
import os
import argparse


def resize2small(image):
    # img_show(image, "img1")
    small = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    # img_show(small, "img2")
    return small


def img_show(img, name=""):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize_image(image, height, width):
    top, bottom, left, right = (0, 0, 0, 0)
    # 获取图片尺寸
    h, w, _ = image.shape
    # 对于长宽不等的图片，找到最长的一边
    longest_edge = max(h, w)
    # 计算短边需要增加多少像素宽度才能与长边等长(相当于padding，长边的padding为0，短边才会有padding)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    # 给图片增加padding，使图片长、宽相等
    # top, bottom, left, right分别是各个边界的宽度，cv2.BORDER_CONSTANT是一种border type，表示用相同的颜色填充
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    # 调整图像大小并返回图像，目的是减少计算量和内存占用，提升训练速度
    return cv2.resize(constant, (height, width))


def norm(img):
    img = img.astype('float32')
    img = (img - img.mean(axis=(0, 1, 2), keepdims=True)) / img.std(axis=(0, 1, 2), keepdims=True)
    img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
    return img


def preprocess_image(array, height, width):
    # img_show(image, "img")
    # img_show(image_normed, "img_normed")
    # 将所有图像加工成长宽定维
    image_resized = resize_image(array, height, width)
    image_normed = norm(image_resized)
    # 将图像的RGB三通道转2d灰度化
    image_preprocessed = cv2.cvtColor(image_normed, cv2.COLOR_RGB2GRAY)
    # img_show(image_preprocessed, "img_preprocessed")
    return image_preprocessed


if __name__ == "__main__":
    # 读取图像
    parser = argparse.ArgumentParser()
    # --------------------------- data path -------------------------#
    parser.add_argument('--file_number', '--file_number', default=50)
    parser.add_argument('--height', '--height', default=280)
    parser.add_argument('--width', '--width', default=280)
    opt = parser.parse_args()
    stack = list()

    img_npy_folder = "G:/image_npy_11/"
    img_info_folder = "F:/SGRAF/f30k_precomp/"
    img_info = img_info_folder + "img.json"
    files = os.listdir(img_npy_folder)
    with open(img_info, "r") as ims:
        img_info_dict = eval(ims.read())
    c = 0
    with open(img_info_folder + "train_caps.txt", "w") as cap:
        for file in files:
            if c > opt.file_number:
                break
            img_path = img_npy_folder + file
            name = file.replace(".npy", ".jpg")
            if name in img_info_dict:
                image = np.load(img_path)
                img = preprocess_image(image, height=opt.height, width=opt.width)
                caption = "".join(img_info_dict[name])
                cap.write(caption)
                cap.write("\n")
                stack.append(img)
                c += 1
            else:
                print(file)
    np.save(img_info_folder + "train_ims.npy", np.array(stack))
    print(len(stack))
