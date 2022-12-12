import argparse
import pathlib
import shutil
import time
from pathlib import Path
import random







def dataset_mak():
    '''
    制作数据集，按照训练集：验证集：测试集=8:1:1的比例自动制作数据集
    采用复制方式，源文件保留
    :return:
    '''
    max_num = opt.max_img_num
    train_size = opt.train_size
    sor_path = opt.path1   # 数据路径，包含各目标文件夹
    sor_p = Path(sor_path)
    path = sor_p.parent   # 数据目录父级路径，用于生成datasets路径
    p = Path(path)
    da_p = 'datasets'
    train_p = p / da_p / 'train'
    valid_p = p / da_p / 'valid'
    test_p = p / da_p / 'test'

    if not train_p.is_dir():
        pathlib.Path(train_p).mkdir(parents=True, exist_ok=True)   # 建立训练集文件夹
        print(f'folder {train_p.stem} ready')
    if not valid_p.is_dir():
        pathlib.Path(valid_p).mkdir(parents=True, exist_ok=True)   # 建立验证集文件夹
        print(f'folder {valid_p.stem} ready')
    if not test_p.is_dir():
        pathlib.Path(test_p).mkdir(parents=True, exist_ok=True)   # 建立测试集文件夹
        print(f'folder {test_p.stem} ready\n')
    print('dataset folder ready\n')



    for f_1 in sor_p.iterdir():   # 遍历源文件夹
        if not (train_p / f_1.name).is_dir():
            pathlib.Path(train_p / f_1.name).mkdir(parents=True, exist_ok=True)   # 在训练集文件夹中建立对应源文件夹的目标文件夹
        if not (valid_p / f_1.name).is_dir():
            pathlib.Path(valid_p / f_1.name).mkdir(parents=True, exist_ok=True)   # 在验证文件夹中建立对应源文件夹的目标文件夹
        if not (test_p / f_1.name).is_dir():
            pathlib.Path(test_p / f_1.name).mkdir(parents=True, exist_ok=True)   # 在测试集文件夹中建立对应源文件夹的目标文件夹
    print('target folder ready\n')



    for targ_folder in (p / da_p).iterdir():
        t0 = time.time()
          # print(targ_folder)
        if targ_folder.stem == 'train':   # 判断条件，防止重复复制，优先写入训练集
            j = 1
            for file_folder in sor_p.iterdir():   # 遍历训练目标文件夹
                print(f'{"="*20} {file_folder.name}({j}/{len(list(sor_p.iterdir()))}) 整理中{"="*20}')
                  #  j = j + 1
                list_img = []
                for img in file_folder.iterdir():   # 遍历训练目标图片
                    seed = random.randint(1, 8)
                    if len(list(file_folder.iterdir())) >= int(max_num):
                        if seed <= 4:
                            list_img.append(img)   # 将图片地址写入空列表，为后续数据切分做准备
                    else:
                        list_img.append(img)
                if file_folder.stem == (train_p / file_folder.name).stem:   # 判断源文件夹中和训练集文件夹中名称是否一致
                      # print(file_folder, (test_p / file_folder.name))
                    random.shuffle(list_img)   # 乱序列表，为随机抽取做准备
                    num1 = int(len(list_img) * train_size)   # 设置切割占比
                    random_sample_list = random.sample(list_img, num1)  # 随机取样num1个元素
                    for i, x in enumerate(random_sample_list):
                        shutil.copy(x, (train_p / file_folder.name))   # 复制列表random_sample_list中的元素
                        lenth = len(list_img[:num1])
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
                    for x2 in random_sample_list:
                        shutil.copy(x2, valid_p / file_folder.name)    # 按比例复制列表
                        lenth = len(list_img[:num2])
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
                    for x3 in list_img:
                        shutil.copy(x3, test_p / file_folder.name)    # 将剩余的图片写入测试集对应文件夹
                        lenth = len(list_img)
                        a = "*" * int((i / lenth) * 35)
                        b = "." * int(35 - ((i / lenth) * 35))
                        c = (i / lenth) * 100
                        i = i + 1
                        print(f'\rtest_dir  进度：{c:.0f}%[{a}->{b}]', end="")

                    print(f'\n{"-"*20} {file_folder.name}({j}/{len(list(sor_p.iterdir()))}) 已完毕{"-"*20}\n\n')
                j = j + 1

    print('====Dataset has been ready====')




if __name__ == '__main__':
    path = Path.cwd()
    p = Path(path)
    parser = argparse.ArgumentParser()
    parser.add_argument("--path1", type=str, default=r'I:\秦岭野生动物园2021.6.26-7.4\小熊猫狒狒数据集\个体识别', help="数据路径")
    parser.add_argument("--train_size", type=int, default=0.8, help="训练集数据占比，验证集与测试集各占剩余数据50%")
    parser.add_argument("--max_img_num", type=int, default=50000, help="目标图片最大数量，若高于此数量将随机删除至此数量")
    opt = parser.parse_args()
    print(opt)
    dataset_mak()