from face_util import face_process
import numpy as np
import config
import caffe
import os
import sys
import cv2

def measure_triangle_sample(file_name):

    triangle = []
    with open(file_name) as file :
        for line in file :
            x,y,z = line.split()
            triangle.append((int(x), int(y), int(z)))
    return triangle
    
def generate_feature(prototxt, caffemodel, register_face_dir, save_txt_dir, save_feature_dir):
    
    if not os.path.isfile(caffemodel):
        print ("caffemodel not found!")
    caffe.set_mode_gpu()
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    imgs = []
    with open(save_txt_dir,'w') as f:
        for img_name in os.listdir(register_face_dir):
            imgs.append(os.path.join(register_face_dir,img_name))
            f.write(img_name.split('.')[0] + '\n')
    feats = np.zeros((len(imgs), 256), dtype = np.float32)
    for i in range(0, len(imgs)):
        input = cv2.imread(imgs[i], 0)
        input = cv2.resize(input, (128, 128), interpolation = cv2.INTER_CUBIC)
        img_blobinp = input[np.newaxis, np.newaxis, :, :] / 255.0
        net.blobs['data'].reshape(*img_blobinp.shape)
        net.blobs['data'].data[...] = img_blobinp
        net.blobs['data'].data.shape
        net.forward()
        feature = net.blobs['eltwise_fc1'].data
        feats[i, :] = feature
    feats.tofile(save_feature_dir)

def crop_template_face(template_Img_path, save_Img_path, min_value=1, max_value=50):

    print 'start crop template face img .....'
    face_process.crop_faces_dir(template_Img_path, save_Img_path, min_value, max_value)
    print 'crop template face img end .....'

def generate_AlignImg_data(imput_TemplateImg_dir, save_AlignImg_dir, save_KeyPoints_dir,min_value=1, max_value=50):
    
    print 'generate AlignImg data .....'

    for img_name in os.listdir(imput_TemplateImg_dir):
        img_path = imput_TemplateImg_dir + img_name
        print ('process: %s' %img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        _, templet_img_align , _ , points = face_detector.align_stable(img, min_value, max_value)

        if points is None:
            continue
        
        save_AliImg_path = os.path.join(save_AlignImg_dir, img_name)
        cv2.imwrite(save_AliImg_path, templet_img_align)
        
        txt_file_name = img_name.split('.')[0] + '.txt'
        save_AliImgPoint_path = os.path.join(save_KeyPoints_dir, txt_file_name)

        f = open(save_AliImgPoint_path, 'w')
        for i in range(0, len(points)):
            f.write(str(points[i][0]) +' ' + str(points[i][1]) + '\n')

def test_data_uniform(align_img_dir, align_txt_dir):

    img_list = os.listdir(align_img_dir)
    txt_list = os.listdir(align_txt_dir)

    for file_name in img_list:
        txt_name = file_name.split('.')[0] + '.txt'
        if txt_name not in txt_list:
            print "img file don't match txt file."
            return None, None

    return img_list, txt_list

def get_AliImg_data(align_img_dir, align_txt_dir):
    
    img_dic = {}
    point_dic={}

    img_list, txt_list = test_data_uniform(align_img_dir, align_txt_dir)
    
    if img_list is None or txt_list is None:
        print 'load aliimg data failed.'
        return None, None

    for file_name in img_list:
        img_name = file_name.split('.')[0]
        txt_name = img_name + '.txt'
        img_path = os.path.join(align_img_dir, file_name)
        txt_path = os.path.join(align_txt_dir, txt_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        img_dic[img_name] = img
        f = open(txt_path, 'r')
        point_list = []
        while True:
            lines = f.readline()
            if not lines:
                break
            y, x = [int(i) for i in lines.split(' ')]
            point_list.append((y, x))

        point_dic[img_name] = point_list

    return img_dic, point_dic

if __name__ == '__main__':
    
    face_detector = face_process.init_detector()
    # #1
    # template_ManImg_path = config.template_ManImg_path
    # save_ManImg_path = config.save_ManImg_path
    # if not os.path.exists(save_ManImg_path):
    #     os.makedirs(save_ManImg_path)
    # crop_template_face(template_ManImg_path, save_ManImg_path)

    # template_feManImg_path = config.template_feManImg_path
    # save_feManImg_path = config.save_feManImg_path
    # if not os.path.exists(save_feManImg_path):
    #     os.makedirs(save_feManImg_path)
    # crop_template_face(template_feManImg_path, save_feManImg_path)
    
    # #2
    # generate_feature(config.prototxt, config.caffemodel, config.save_ManImg_path, config.save_male_txt, config.save_male_feature)
    # generate_feature(config.prototxt, config.caffemodel, config.save_feManImg_path, config.save_female_txt, config.save_female_feature)
    
    #3
    # if not os.path.exists(config.save_AlignFemaleImg_dir):
    #     os.makedirs(config.save_AlignFemaleImg_dir)
    # if not os.path.exists(config.save_AlignMaleImg_dir):
    #     os.makedirs(config.save_AlignMaleImg_dir)
    # if not os.path.exists(config.save_AlignFemalePoint_dir):
    #     os.makedirs(config.save_AlignFemalePoint_dir)
    # if not os.path.exists(config.save_AlignMalePoint_dir):
    #     os.makedirs(config.save_AlignMalePoint_dir)
    # generate_AlignImg_data(config.template_feManImg_path, config.save_AlignFemaleImg_dir, config.save_AlignFemalePoint_dir,min_value=1, max_value=50)
    # generate_AlignImg_data(config.template_ManImg_path, config.save_AlignMaleImg_dir, config.save_AlignMalePoint_dir,min_value=1, max_value=50)
    
    #
    # img_dic, point_dic = get_AliImg_data(config.save_AlignFemaleImg_dir, config.save_AlignFemalePoint_dir)

    # for key in img_dic.keys():
    #     face_process.show_landmarks(img_dic[key], point_dic[key])
    

    ##---------------------------------------------------
    generate_AlignImg_data(config.template_BabyImg_path, config.save_AlignBabyImg_dir, config.save_AlignBabyPoint_dir,min_value=1, max_value=50)
    img_dic, point_dic = get_AliImg_data(config.save_AlignBabyImg_dir, config.save_AlignBabyPoint_dir)

    for key in img_dic.keys():
        face_process.show_landmarks(img_dic[key], point_dic[key])
