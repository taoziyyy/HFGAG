import json
import os

import pandas as pd

# 将.conll文件转化为.txt文件
def gen_train_data(file_path, save_path):

    path = file_path
    lineList = []
    file = open(path, "r", encoding='utf-8')  # 以只读模式读取文件
    while 1:
        line = file.readline()
        if not line:
            # print("End or Error.")
            break
        reline = line.replace('-X-', '')
        reline = reline.replace('-X-_','')
        reline = reline.replace('_', '')

        lineList.append(reline)

    file.close()
    file = open(save_path, 'w', encoding='utf-8')
    for i in lineList:
        file.write(i)
    file.close()


def main():
    folder = os.path.exists('data/hairFollicle_BIO')  # 创建一个新的文件夹目录放置转换为BIO格式的文件
    if not folder:
        os.makedirs('data/hairFollicle')
    gen_train_data('data/hairFollicle/train.conll', 'data/hairFollicle_BIO/train.txt')
    gen_train_data('data/hairFollicle/test.conll', 'data/hairFollicle_BIO/test.txt')
    gen_train_data('data/hairFollicle/dev.conll', 'data/hairFollicle_BIO/dev.txt')


if __name__ == '__main__':
    main()
