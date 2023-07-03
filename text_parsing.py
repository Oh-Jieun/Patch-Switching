#folder_move
from distutils import filelist
import os #파일명, 폴더명 정보를 읽어오기 위한 모듈
import shutil #파일 이동을 위한 모듈
import numpy as np
import os

# set directory
base_dir = r"C:\Users\KETI\PycharmProjects\pythonProject1\tiny-imagenet-200\val"

path_after = r"C:\Users\KETI\PycharmProjects\pythonProject1\tiny-imagenet-200\val"

file_name = 'val_annotations.txt'
path = os.path.join(base_dir, file_name)

category = []
dict_list = []
ms_list = []
dict={}

file_opened = open(path)

def fileList(path_before : str)->list :
    file_list = os.listdir(path_before) 

    category = [] 
    for line in file_opened.readlines():
        temp_list = line.split("\t")
        category.append(temp_list[1])
        dict_list.append(temp_list[0])
        ms_list.append(temp_list[0:2])
    
    temp_set = set(category) 
    result = list(temp_set) 

    for value in ms_list:
        shutil.move(path_before+"/"+value[0], path_after+"/"+value[1]+"/"+value[0])

    return result #결과 리턴

def makeFolder(path_after : str, file_list : list):    
    for file in file_list:
        try:
            os.makedirs(path_after+"/"+file)
        except:
            pass
    
    for line in file_opened.readlines():
        print(line)
        temp_list = line.split("\t")
        dict.append(temp_list[0])
    print(dict)
    
    for key, value in dict.items():
        shutil.move(path_before+"/"+key, path_after+"/"+value)
    
if __name__ == "__main__" :
    path_before = r"C:\Users\KETI\PycharmProjects\pythonProject1\tiny-imagenet-200\val\images"
    file_list = fileList(path_before)

    path_after = r"C:\Users\KETI\PycharmProjects\pythonProject1\tiny-imagenet-200\val"
    makeFolder(path_after, file_list)