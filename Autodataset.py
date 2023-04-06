import argparse
import shutil
import sys
import time
import random
import re
import os
import torch
import torchvision
from PIL import Image
import cv2
import numpy as np
from pathlib import Path as p
from tqdm import tqdm
import yaml





class AutoDataset:
    """
    inputpath: data source path.
    outputpath: output path of processed data.
    config_file: config file path.
    train_size: split data to dataset, if train size is 0.8, then val size and test size
                will be 0.1 and 0.1 each.
    max_size: max number of files in dataset, default is 500000
    datasetname: new dataset name after process
    """

    def __init__(self, inputpath, outputpath, config_file,
                 train_size=0.8, max_size=500000, datasetname='AutoDataset'):
        self.inputpath = inputpath
        if outputpath is not None:
            self.outputpath = outputpath
        else:
            self.outputpath = p(inputpath).parent / f'{p(inputpath).stem}_output'
        self.config_file = config_file
        self.train_size = train_size
        self.max_size = max_size
        self.datasetname = datasetname

    def makedataset(self, datasettype='individual', convert=False, enhance=False):
        """
        datasettype: choice in ['specie', 'individual'], 'specie' for making dataset
                    with yolo format and 'individual' for resnet format.
        convert: whether convert data to standard name format, only make resnet format dataset works.
                for example: '多多' convert to 'Chimpanzee_000000000000_多多'
        """
        type = ['specie', 'individual']

        if datasettype not in type:
            raise NameError(f"Only 'specie' or 'individual' type supported, please check datasettype")
        elif datasettype == type[0]:
            if enhance:
                self.imgenhance_yolo(self.inputpath, self.outputpath)
                self.dataset_maker_yolo(self.outputpath, self.train_size,
                                        self.max_size, self.datasetname)
            else:
                self.dataset_maker_yolo(self.inputpath, self.train_size,
                                        self.max_size, self.datasetname)
        elif datasettype == type[1]:
            if convert:
                self.convert_std_filename(self.inputpath)
                if enhance:
                    self.imgenhance_resnet(self.inputpath, self.outputpath)
                    self.dataset_maker(self.outputpath, self.train_size,
                                       self.max_size, self.datasetname)
                else:
                    self.dataset_maker(self.inputpath, self.train_size,
                                       self.max_size, self.datasetname)
            else:
                if enhance:
                    self.imgenhance_resnet(self.inputpath, self.outputpath)
                    self.dataset_maker(self.outputpath, self.train_size,
                                       self.max_size, self.datasetname)
                else:
                    self.dataset_maker(self.inputpath, self.train_size,
                                       self.max_size, self.datasetname)

    def convert_std_filename(self, sor_p, init_num=0):
        print('coverting files\n')
        sp_ind = self.load_config(self.config_file)['sp_ind']
        sor_p = p(sor_p)
        pattern = re.compile('\d')
        for f_1 in sor_p.iterdir():  # 遍历源文件夹名, 检查是否已转换
            check_list = []
            oringin_num = pattern.findall(str(f_1.stem))
            if len(oringin_num) == 12:
                check_list.append(f_1.stem)
                info = 'seems folders name have been converted, please check it'
                raise NameError(info, check_list)
        for f_1 in sor_p.iterdir():  # 遍历源文件夹
            oringin_num = pattern.findall(str(f_1.stem))
            if len(oringin_num) != 12:
                temp = sp_ind.values()
                for key, values in sp_ind.items():
                    if f_1.stem in values:
                        sp_name = key
                        standard_new_name = f'{sp_name}_{init_num:012d}_{f_1.name}'
                    # standard_new_name = f'{init_num:012d}_{f_1.name}'
                    # else:
                    #     raise NameError(f'{f_1.stem} is not belong to any species,'
                    #                     f'please check config file. ')
                        f_1.rename(f_1.parent / standard_new_name)
                        init_num = init_num + 1

    def imgenhance_resnet(self, imgpath, output, seed=17):
        print('enhancing images\n')
        torch.manual_seed(seed)
        imgpath_p = p(imgpath)
        output_p = p(output)

        a = [torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
             # torchvision.transforms.Grayscale(num_output_channels=1),
             torchvision.transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
             # torchvision.transforms.RandomInvert(p=0.5),
             torchvision.transforms.RandomPosterize(4, p=0.5),
             # torchvision.transforms.RandomSolarize(400, p=0.5),
             torchvision.transforms.RandomAdjustSharpness(3, p=0.5),
             torchvision.transforms.RandomHorizontalFlip(p=0.5),
             torchvision.transforms.RandomVerticalFlip(p=0.5),
             torchvision.transforms.RandomRotation(degrees=90),
             torchvision.transforms.RandomRotation(degrees=120),
             torchvision.transforms.RandomRotation(degrees=150),
             # torchvision.transforms.RandomAutocontrast(p=0.5),
             torchvision.transforms.RandomEqualize(p=0.5)]
        temp = list(imgpath_p.iterdir())

        for dir in imgpath_p.iterdir():
            if not (output_p / dir.stem).exists():
                (output_p / dir.stem).mkdir(parents=True)
            init_num = 0
            for bhv in tqdm(list(dir.iterdir()), postfix=f'processing [{dir.stem}] files'):
                if bhv.suffix == '.jpg' or '.JPG' or '.png' or '.PNG'\
                                 '.jpeg' or '.JPEG':
                    img = cv2.imread(str(bhv))
                    # temp = f'{output}/{dir.stem}/{init_num:012d}.jpg'
                    origin_img_name = str(bhv)
                    new_img_name = f'{output}/{dir.stem}/{init_num:012d}.jpg'
                    shutil.copy(origin_img_name, new_img_name)
                    init_num2 = init_num + 1
                    for i, trans in enumerate(a):
                        trans_im = trans(Image.fromarray(img))
                        trans_im = np.array(trans_im)
                        cv2.imwrite(f'{output}/{dir.stem}/{init_num2:012d}.jpg', trans_im)
                        init_num2 = init_num2 + 1

                    init_num = init_num2 + 1

    def imgenhance_yolo(self, imgpath, output, seed=17):
        print('enhancing images\n')
        torch.manual_seed(seed)
        imgpath_p = p(imgpath)
        output_p = p(output)

        a = [torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
             torchvision.transforms.Grayscale(num_output_channels=1),
             torchvision.transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
             # torchvision.transforms.RandomInvert(p=0.5),
             torchvision.transforms.RandomPosterize(4, p=0.5),
             # torchvision.transforms.RandomSolarize(400, p=0.5),
             torchvision.transforms.RandomAdjustSharpness(3, p=0.5),
             # torchvision.transforms.RandomAutocontrast(p=0.5),
             torchvision.transforms.RandomEqualize(p=0.5)]

        init_num = 0

        for dir in imgpath_p.iterdir():
            if not (output_p / dir.stem).exists():
                (output_p / dir.stem).mkdir(parents=True)
        for dir in imgpath_p.iterdir():
            for bhv in tqdm(list(dir.iterdir()), postfix=f'processing [{dir.stem}] files'):
                if bhv.suffix == '.jpg' or '.JPG' or '.png' or '.PNG'\
                                 '.jpeg' or '.JPEG':
                    img = cv2.imread(str(bhv))
                    # temp = f'{output}/{dir.stem}/{init_num:012d}.jpg'
                    origin_img_name = str(bhv)
                    new_img_name = f'{output}/{dir.stem}/{init_num:012d}.jpg'
                    shutil.copy(origin_img_name, new_img_name)
                    origin_jsn_name = bhv.parent.parent / 'labels' / (bhv.stem + '.txt')
                    new_jsn_name = output_p / 'labels'/ f'{init_num:012d}.txt'
                    shutil.copy(origin_jsn_name, new_jsn_name)
                    init_num2 = init_num + 1
                    for i, trans in enumerate(a):
                        trans_im = trans(Image.fromarray(img))
                        trans_im = np.array(trans_im)
                        cv2.imwrite(f'{output}/{dir.stem}/{init_num2:012d}.jpg', trans_im)
                        trans_new_jsn_name = output_p / 'labels' / f'{init_num2:012d}.txt'
                        shutil.copy(origin_jsn_name, trans_new_jsn_name)
                        init_num2 = init_num2 + 1

                    init_num = init_num2 + 1

    def dataset_maker(self, inputpath, train_size, max_size, datasetname):
        '''
        制作数据集，按照训练集：验证集：测试集=8:1:1的比例自动制作数据集
        采用复制方式，源文件保留
        :return:
        '''

        max_num = max_size
        sor_p = p(inputpath)  # 数据路径，包含各目标文件夹
        parent_path = sor_p.parent   # 数据目录父级路径，用于生成datasets路径
        da_p = f'{sor_p.stem}_{datasetname}'
        train_p = parent_path / da_p / 'train'
        valid_p = parent_path / da_p / 'val'
        test_p = parent_path / da_p / 'test'

        if not train_p.is_dir():
            p(train_p).mkdir(parents=True, exist_ok=True)   # 建立训练集文件夹
            print(f'folder {train_p.stem} ready')
        if not valid_p.is_dir():
            p(valid_p).mkdir(parents=True, exist_ok=True)   # 建立验证集文件夹
            print(f'folder {valid_p.stem} ready')
        if not test_p.is_dir():
            p(test_p).mkdir(parents=True, exist_ok=True)   # 建立测试集文件夹
            print(f'folder {test_p.stem} ready\n')
        print('dataset folder ready\n')


        for f_1 in sor_p.iterdir():   # 遍历源文件夹
            if not (train_p / f_1.name).is_dir():
                p(train_p / f_1.name).mkdir(parents=True, exist_ok=True)   # 在训练集文件夹中建立对应源文件夹的目标文件夹
            if not (valid_p / f_1.name).is_dir():
                p(valid_p / f_1.name).mkdir(parents=True, exist_ok=True)   # 在验证文件夹中建立对应源文件夹的目标文件夹
            if not (test_p / f_1.name).is_dir():
                p(test_p / f_1.name).mkdir(parents=True, exist_ok=True)   # 在测试集文件夹中建立对应源文件夹的目标文件夹
        print('target folder ready\n')



        for targ_folder in (parent_path / da_p).iterdir():
            t0 = time.time()
              # print(targ_folder)
            if targ_folder.stem == 'train':   # 判断条件，防止重复复制，优先写入训练集
                j = 1
                for file_folder in sor_p.iterdir():   # 遍历训练目标文件夹
                    print(f'{"="*20} {file_folder.name}({j}/{len(list(sor_p.iterdir()))}) 整理中{"="*20}')
                      #  j = j + 1
                    list_img = list(file_folder.iterdir())
                    if len(list_img) >= int(max_num):
                        list_img = random.sample(list_img, max_num)


                    # list_img = []
                    # for img in file_folder.iterdir():   # 遍历训练目标图片
                    #     seed = random.randint(1, 8)
                    #     if len(list(file_folder.iterdir())) >= int(max_num):
                    #         if seed <= 4:
                    #             list_img.append(img)   # 将图片地址写入空列表，为后续数据切分做准备
                    #     else:
                    #         list_img.append(img)
                    if file_folder.stem == (train_p / file_folder.name).stem:   # 判断源文件夹中和训练集文件夹中名称是否一致
                          # print(file_folder, (test_p / file_folder.name))
                        random.shuffle(list_img)   # 乱序列表，为随机抽取做准备
                        num1 = int(len(list_img) * train_size)   # 设置切割占比
                        random_sample_list = random.sample(list_img, num1)  # 随机取样num1个元素
                        lenth = len(list_img[:num1])
                        for i, x in enumerate(random_sample_list):
                            shutil.copy(x, (train_p / file_folder.name))   # 复制列表random_sample_list中的元素
                            # lenth = len(list_img[:num1])
                            a = "*" * int(((i+1) / lenth) * 35)
                            b = "." * int(35 - (((i+1) / lenth) * 35))
                            c = (i / lenth) * 100
                            print(f'\rtrain_dir 进度：{c:.0f}%[{a}->{b}]', end="")
                            list_img.remove(x)  # 在list_img中删除random_sample_list存在的元素


                    if file_folder.stem == (valid_p / file_folder.name).stem:   # 判断源文件夹中和验证集文件夹中名称是否一致
                        print('\n')
                        random.shuffle(list_img)    # 乱序列表，为随机抽取做准备
                        num2 = int(len(list_img) * 0.5)    # 剩余数据50%
                        random_sample_list = random.sample(list_img, num2)
                        i = 1
                        lenth = len(list_img[:num2])
                        for x2 in random_sample_list:
                            shutil.copy(x2, valid_p / file_folder.name)    # 按比例复制列表
                            # lenth = len(list_img[:num2])
                            a = "*" * int((i / lenth) * 35)
                            b = "." * int(35 - ((i / lenth) * 35))
                            c = (i / lenth) * 100
                            i = i + 1
                            print(f'\rvalid_dir 进度：{c:.0f}%[{a}->{b}]', end="")
                            list_img.remove(x2) # 在list_img中删除random_sample_list存在的元素
                              #  time.sleep(0.1)


                    if file_folder.stem == (test_p / file_folder.name).stem:    # 判断源文件夹中和测试集文件夹中名称是否一致
                        print('\n')
                        i = 1
                        lenth = len(list_img)
                        for x3 in list_img:
                            shutil.copy(x3, test_p / file_folder.name)    # 将剩余的图片写入测试集对应文件夹
                            # lenth = len(list_img)
                            a = "*" * int((i / lenth) * 35)
                            b = "." * int(35 - ((i / lenth) * 35))
                            c = (i / lenth) * 100
                            i = i + 1
                            print(f'\rtest_dir  进度：{c:.0f}%[{a}->{b}]', end="")

                        print(f'\n{"-"*20} {file_folder.name}({j}/{len(list(sor_p.iterdir()))}) 已完毕{"-"*20}\n\n')
                    j = j + 1

        print('====Dataset has been ready====')

    def dataset_maker_yolo(self, inputpath, train_size, max_size, datasetname):
        '''
        制作数据集，按照训练集：验证集：测试集=8:1:1的比例自动制作数据集
        采用复制方式，源文件保留
        :return:
        '''
        max_num = max_size
        sor_p = p(inputpath)  # 数据路径，包含各目标文件夹
        parent_path = sor_p.parent  # 数据目录父级路径，用于生成datasets路径
        da_p = f'{sor_p.stem}_{datasetname}'
        train_p = parent_path / da_p / 'train'
        valid_p = parent_path / da_p / 'val'
        test_p = parent_path / da_p / 'test'

        if not train_p.is_dir():
            p(train_p).mkdir(parents=True, exist_ok=True)  # 建立训练集文件夹
            print(f'folder {train_p.stem} ready')
        if not valid_p.is_dir():
            p(valid_p).mkdir(parents=True, exist_ok=True)  # 建立验证集文件夹
            print(f'folder {valid_p.stem} ready')
        if not test_p.is_dir():
            p(test_p).mkdir(parents=True, exist_ok=True)  # 建立测试集文件夹
            print(f'folder {test_p.stem} ready\n')
        print('dataset folder ready\n')

        for f_1 in sor_p.iterdir():  # 遍历源文件夹
            if not (train_p / f_1.name).is_dir():
                p(train_p / f_1.name).mkdir(parents=True, exist_ok=True)  # 在训练集文件夹中建立对应源文件夹的目标文件夹
            if not (valid_p / f_1.name).is_dir():
                p(valid_p / f_1.name).mkdir(parents=True, exist_ok=True)  # 在验证文件夹中建立对应源文件夹的目标文件夹
            if not (test_p / f_1.name).is_dir():
                p(test_p / f_1.name).mkdir(parents=True, exist_ok=True)  # 在测试集文件夹中建立对应源文件夹的目标文件夹
        print('target folder ready\n')

        for targ_folder in (parent_path / da_p).iterdir():
            t0 = time.time()
            # print(targ_folder)
            if targ_folder.stem == 'train':  # 判断条件，防止重复复制，优先写入训练集
                j = 1
                for file_folder in sor_p.iterdir():  # 遍历训练目标文件夹
                    if file_folder.stem == 'images':
                        print(f'{"=" * 20} {file_folder.name}({j}/{len(list(sor_p.iterdir()))}) 整理中{"=" * 20}')
                        #  j = j + 1
                        list_img = list(file_folder.iterdir())
                        if len(list_img) >= int(max_num):
                            list_img = random.sample(list_img, max_num)
                        if file_folder.stem == (train_p / file_folder.name).stem:  # 判断源文件夹中和训练集文件夹中名称是否一致
                            # print(file_folder, (test_p / file_folder.name))
                            random.shuffle(list_img)  # 乱序列表，为随机抽取做准备
                            num1 = int(len(list_img) * train_size)  # 设置切割占比
                            random_sample_list = random.sample(list_img, num1)  # 随机取样num1个元素
                            lenth = len(list_img[:num1])
                            for i, x in enumerate(random_sample_list):
                                shutil.copy(x, (train_p / file_folder.name))  # 复制列表random_sample_list中的元素
                                shutil.copy(f'{x.parent.parent}/labels/{x.stem}.txt',
                                            (train_p / 'labels'))  # 复制列表random_sample_list中的元素

                                a = "*" * int(((i + 1) / lenth) * 35)
                                b = "." * int(35 - (((i + 1) / lenth) * 35))
                                c = (i / lenth) * 100
                                print(f'\rtrain_dir 进度：{c:.0f}%[{a}->{b}]', end="")
                                list_img.remove(x)  # 在list_img中删除random_sample_list存在的元素

                        if file_folder.stem == (valid_p / file_folder.name).stem:  # 判断源文件夹中和验证集文件夹中名称是否一致
                            print('\n')
                            random.shuffle(list_img)  # 乱序列表，为随机抽取做准备
                            num2 = int(len(list_img) * 0.5)  # 剩余数据50%
                            random_sample_list = random.sample(list_img, num2)
                            i = 1
                            lenth = len(list_img[:num2])
                            for x2 in random_sample_list:
                                shutil.copy(x2, valid_p / file_folder.name)  # 按比例复制列表
                                shutil.copy(f'{x2.parent.parent}/labels/{x2.stem}.txt', (valid_p / 'labels'))

                                a = "*" * int((i / lenth) * 35)
                                b = "." * int(35 - ((i / lenth) * 35))
                                c = (i / lenth) * 100
                                i = i + 1
                                print(f'\rvalid_dir 进度：{c:.0f}%[{a}->{b}]', end="")
                                list_img.remove(x2)  # 在list_img中删除random_sample_list存在的元素
                                #  time.sleep(0.1)

                        if file_folder.stem == (test_p / file_folder.name).stem:  # 判断源文件夹中和测试集文件夹中名称是否一致
                            print('\n')
                            i = 1
                            lenth = len(list_img)
                            for x3 in list_img:
                                shutil.copy(x3, test_p / file_folder.name)  # 将剩余的图片写入测试集对应文件夹
                                shutil.copy(f'{x3.parent.parent}/labels/{x3.stem}.txt', (test_p / 'labels'))

                                a = "*" * int((i / lenth) * 35)
                                b = "." * int(35 - ((i / lenth) * 35))
                                c = (i / lenth) * 100
                                i = i + 1
                                print(f'\rtest_dir  进度：{c:.0f}%[{a}->{b}]', end="")

                            print(
                                f'\n{"-" * 20} {file_folder.name}({j}/{len(list(sor_p.iterdir()))}) 已完毕{"-" * 20}\n\n')
                        j = j + 1

        print('====Dataset has been ready====')

    def load_config(self, filepath):
        with open(filepath, encoding='UTF-8') as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)

        return data_dict


