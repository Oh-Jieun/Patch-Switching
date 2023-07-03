from PIL import Image
from itertools import product
import os
import matplotlib.pyplot as plt
import random

imgPath = root=r"C:\Users\KETI\PycharmProjects\pythonProject1\tiny-imagenet-200-da\train"
savePath = root=r"C:\Users\KETI\PycharmProjects\pythonProject1\tiny-imagenet-200-da\train"


file_lst = os.listdir(imgPath)
testList = []

def patch_switching(filename, imgPath, savePath, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(imgPath, filename))
    w, h = img.size
    temp = []
    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        out = os.path.join(savePath, f'{name}_{i}_{j}{ext}')
        temp.append(out)
        img.crop(box).save(out)
        

    i = 0
    merged = Image.new('RGB', (d*(w//d), d*(h//d)))
    random.shuffle(temp)
    for y in range(h//d):
        for x in range(w//d):
            im = Image.open(temp[i])
            i = i + 1
            merged.paste(im, (d*x, d*y))
        merged.save(f'{name}_da{ext}')
    #이미지 크기가 64*64이므로 4분할 하기 위한 코드
    path1 = os.path.join(savePath, f'{name}_{0}_{0}{ext}')
    path2 = os.path.join(savePath, f'{name}_{0}_{32}{ext}')
    path3 = os.path.join(savePath, f'{name}_{32}_{0}{ext}')
    path4 = os.path.join(savePath, f'{name}_{32}_{32}{ext}')
    os.remove(path1)
    os.remove(path2)
    os.remove(path3)
    os.remove(path4)


for i in range(len(file_lst)):
    str = imgPath + '/' + file_lst[i]
    imgList = os.listdir(str)
    imgList = [os.path.join(str, file) for file in imgList]
    imgList = [file for file in imgList if file.endswith(".jpg") or file.endswith(".png")]
    for cnt, file in enumerate(imgList):
        img = patch_switching(file, str, str, 16)
