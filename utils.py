import os
import numpy as np
import h5py
import cv2 as cv
import random

def make_data(input1, input2, label):
    savepath = './checkpoint/train.h5'
    with h5py.File(savepath ,'w') as hf:
        hf.create_dataset('input1', data=input1)
        hf.create_dataset('input2', data=input2)
        hf.create_dataset('label',data=label)

def read_mask(mask_path): # mask_path: ./mask/
    mask_file_path = []
    mask_img_sequence = []
    mask_file_path =getfilename(mask_path, '.tif')
    for mask_file in mask_file_path:
        mask_img = cv.imread(mask_file, cv.IMREAD_UNCHANGED)
        mask_img_sequence.append(mask_img)


    return mask_img_sequence

def getfilename(path, suffix):
    """ 获取指定目录下的所有指定后缀的文件名 """
    file_list = []
    f_list = os.listdir(path)
    f_list.sort(key=lambda x: int(x[:-4]))
    # print f_list
    for file in f_list:
        # os.path.splitext():分离文件名与扩展名
        if os.path.splitext(file)[1] == suffix:
            file_list.append(os.path.join(path, file))
    return file_list

def getdir(path):
    """  acquire folder names """
    file_dir = []
    dirs = os.listdir(path)
    for dir in dirs:
        file_dir.append(os.path.join(path, dir))
    return file_dir

def test_setup(data_path): #data_path : './test/'  prepare test data
    mask_dir = './mask/'
    test_in1 = './test/input1'
    test_in2 = './test/input2'
    file_name = getfilename(data_path, '.tif')
    sub_mask_sequence = read_mask(mask_dir)
    ori_image_sequence = []
    image_size = 64
    stride = 30
    valid_count = 0
    for name in file_name:
        tmp_img = cv.imread(name, cv.IMREAD_UNCHANGED)
        ori_image_sequence.append(tmp_img)
    for k in range(len(ori_image_sequence)-1):
        tmp_input1 = ori_image_sequence[k]
        tmp_input2 = ori_image_sequence[k + 1]
        h_1, w_1 = tmp_input1.shape
        h_2, w_2 = tmp_input2.shape
        for x in range(0, h_1 - image_size + 1, stride):
            for y in range(0, w_1 - image_size + 1, stride):
                sub_origin = tmp_input1[x:x + image_size, y:y + image_size]
                sub_input2 = tmp_input2[x:x + image_size, y:y + image_size]
                all_1 = (sub_origin > 0).all()
                # judge if the image exist zero
                if all_1:
                    valid_count = valid_count + 1
                    # add mask
                    sub_input1 = sub_mask_sequence[1]*sub_origin

                    # normalized
                    sub_input1 = sub_input1 / 80.0
                    sub_input2 = sub_input2 / 80.0
                    sub_origin = sub_origin / 80.0

                    # sub_lable = sub_origin - sub_input1

                    input1_path = os.path.join(test_in1, 'crop%d.tif'%(valid_count))
                    input2_path = os.path.join(test_in2, 'crop%d.tif'%(valid_count))
                    if valid_count == 1:
                        cv.imwrite('origin.tif', sub_origin)
                    cv.imwrite(input1_path, sub_input1)
                    cv.imwrite(input2_path, sub_input2)

def input_prepare(dataset_path): #dataset_path: './dataset/
    image_size = 40
    t2_subori = 'E:/MSG-experiment/201007/06-12/dataset1h/origin2/' # 需要修改
    t1_subdir = 'E:/MSG-experiment/201007/06-12/dataset1h/temporal1/' # 需要修改
    t2_subdir = 'E:/MSG-experiment/201007/06-12/dataset1h/temporal2/' # 需要修改
    #masks_subdir = 'E:/lstdataset/maskseries/'
    sub_input1_sequence = []
    sub_input2_sequence = []
    sub_label_sequence = []
    t1_dir = dataset_path + t1_subdir
    t2_dir = dataset_path + t2_subdir
    t2_ori_dir = dataset_path + t2_subori
    #masks_dir = dataset_path + masks_subdir
    t1_names = getfilename(t1_dir, '.tif')
    t2_names = getfilename(t2_dir, '.tif')
    t2_oris = getfilename(t2_ori_dir, '.tif')
    #masks = getfilename(masks_dir, '.tif')
    sum = len(t1_names)
    for num in range(sum):
        t2_ori = t2_oris[num]
        t1_name = t1_names[num]
        t2_name = t2_names[num]
        #mask_name = masks[num]
        t2_ori_img = cv.imread(t2_ori, cv.IMREAD_UNCHANGED)
        t1_img = cv.imread(t1_name, cv.IMREAD_UNCHANGED)
        t2_img = cv.imread(t2_name, cv.IMREAD_UNCHANGED)
        #mask_img = cv.imread(mask_name, cv.IMREAD_UNCHANGED)

        label = t2_ori_img - t2_img

        label = label
        t1_img = t1_img
        t2_img = t2_img

        label= label.reshape([image_size, image_size, 1])
        t1_img = t1_img.reshape([image_size, image_size, 1])
        t2_img = t2_img.reshape([image_size, image_size, 1])

        sub_input1_sequence.append(t1_img)
        sub_input2_sequence.append(t2_img)
        sub_label_sequence.append(label)
    # random select input and label
    input12_lablel = list(zip(sub_input1_sequence, sub_input2_sequence,
                              sub_label_sequence))
    random.shuffle(input12_lablel)
    sub_input1_sequence[:], sub_input2_sequence[:], \
    sub_label_sequence[:] = zip(*input12_lablel)

    arrinput1 = np.asarray(sub_input1_sequence)
    arrinput2 = np.asarray(sub_input2_sequence)
    arrlabel = np.asarray(sub_label_sequence)

    make_data(arrinput1, arrinput2, arrlabel)



def input_setup(data_path):  #data_path : './Modis/', prepare train data
    file_dir = getdir(data_path)
    mask_dir = './mask/'
    image_size = 64 # label_size = image_size
    stride = 30 # can be set in config
    sub_input1_sequence = []
    sub_input2_sequence = []
    sub_lable_sequence = []
    sub_valid_squence =[]
    sum = 0
    mask_flag_num = 0
    sub_mask_sequence = read_mask(mask_dir)
    sub_mask = sub_mask_sequence[0]
    for dir in file_dir:
        mask_flag_num = mask_flag_num + 1
        dir = dir + '/'
        file_name = []
        ori_image_sequence = []
        file_name = getfilename(dir, '.tif')
        for name in file_name:
            tmp_img = cv.imread(name, cv.IMREAD_UNCHANGED)  # 直接读取16位-UNCHANGED
            sum =sum + 1
            ori_image_sequence.append(tmp_img)
        for i in range(len(ori_image_sequence)-1):
            sub_valid = 0
            tmp_input1 = ori_image_sequence[i]
            tmp_input2 = ori_image_sequence[i+1]
            h_1, w_1 = tmp_input1.shape
            h_2, w_2 = tmp_input2.shape
            for x in range(0, h_1-image_size+1, stride):
                for y in range(0, w_1-image_size+1, stride):
                    sub_origin = tmp_input1[x:x+image_size, y:y+image_size]
                    sub_input2 = tmp_input2[x:x+image_size, y:y+image_size]

                    all_1 = (sub_origin > 0).all()
                    # judge if the image exist zero
                    if all_1:
                       sub_valid =sub_valid + 1
                       # add mask
                       # if mask_flag_num == 1 or mask_flag_num ==2:
                       #     sub_input1 = sub_mask_sequence[0]*sub_origin
                       # elif mask_flag_num == 3 or mask_flag_num == 4 or\
                       #         mask_flag_num == 5:
                       sub_input1 = sub_mask_sequence[1]*sub_origin
                       # elif mask_flag_num == 6 or mask_flag_num ==7:
                       #     sub_input1 = sub_mask_sequence[2] *sub_origin
                       # elif mask_flag_num == 8 or mask_flag_num == 9 or \
                       #         mask_flag_num == 10:
                       #     sub_input1 = sub_mask_sequence[3] *sub_origin
                       # else:
                       #     sub_input1 = sub_mask_sequence[4] *sub_origin

                       # normalized
                       sub_input1 = sub_input1 / 80.0
                       sub_input2 = sub_input2 / 80.0
                       sub_origin = sub_origin / 80.0

                       sub_lable = sub_origin - sub_input1

                       if sub_valid == 1:
                            cv.imwrite('origin.tif',sub_origin)

                       sub_input1 = sub_input1.reshape([image_size, image_size, 1])
                       sub_input2 = sub_input2.reshape([image_size, image_size, 1])
                       sub_lable = sub_lable.reshape([image_size, image_size, 1])
                       sub_origin = sub_origin.reshape([image_size, image_size, 1])

                       sub_input1_sequence.append(sub_input1)
                       sub_input2_sequence.append(sub_input2)
                       sub_lable_sequence.append(sub_lable)

            sub_valid_squence.append(sub_valid)
    input12_lablel = list(zip(sub_input1_sequence, sub_input2_sequence,
                              sub_lable_sequence))
    random.shuffle(input12_lablel)
    sub_input1_sequence[:], sub_input2_sequence[:], \
    sub_lable_sequence[:] = zip(*input12_lablel)

    arrinput1 = np.asarray(sub_input1_sequence)
    arrinput2 = np.asarray(sub_input2_sequence)
    arrlable = np.asarray(sub_lable_sequence)



    make_data(arrinput1, arrinput2, arrlable)


def read_data(path):
    """
    Read h5 format data file
    Args:
      path: file path of desired file
      data: '.h5' file format that contains train data values
      label: '.h5' file format that contains train label values
    """
    with h5py.File(path, 'r') as hf:
        input1 = np.array(hf.get('input1'))
        input2 = np.array(hf.get('input2'))
        label = np.array(hf.get('label'))
        return input1, input2, label

# test python code
if __name__ == '__main__':
   path = ''
   input_prepare(path)

#     done