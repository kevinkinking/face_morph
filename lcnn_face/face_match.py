import os
import numpy as np
import cv2
import caffe
import scipy
from timer import Timer
import init_data
import config

def init_model(prototxt, caffemodel):
    if not os.path.isfile(caffemodel):
        print ("caffemodel not found!")
        return False
    if not os.path.isfile(prototxt):
        print ("prototxt not found!")
        return False
    caffe.set_mode_cpu()
    global net
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    if net != None:
        return True
    else:
        print ("load model is failed, but prototxt and caffemodel is exit!")
        return False
def init_template_data():
    global features_female_template, features_female_ids
    global features_male_template, features_male_ids
    global female_img_dic, female_points_dic
    global  male_img_dic, male_points_dic
    features_female_template, features_female_ids = read_features(config.save_female_txt, config.save_female_feature)
    features_male_template, features_male_ids = read_features(config.save_male_txt, config.save_male_feature)
    female_img_dic, female_points_dic = init_data.get_AliImg_data(config.save_AlignFemaleImg_dir, config.save_AlignFemalePoint_dir)
    male_img_dic, male_points_dic = init_data.get_AliImg_data(config.save_AlignMaleImg_dir, config.save_AlignMalePoint_dir)

def init_FaceSwitch_data():
    global baby_img_dic, baby_points_dic
    baby_img_dic, baby_points_dic = init_data.get_AliImg_data(config.save_AlignBabyImg_dir, config.save_AlignBabyPoint_dir)

def get_feature(img):
    if img.shape[-1] == 3:
        input = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.shape[-1] == 4:
        input = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        print 'input image channels must in (3, 4)'
        return
    input = cv2.resize(input, (128, 128), interpolation = cv2.INTER_CUBIC)
    img_blobinp = input[np.newaxis, np.newaxis, :, :] / 255.0
    net.blobs['data'].reshape(*img_blobinp.shape)
    net.blobs['data'].data[...] = img_blobinp
    net.blobs['data'].data.shape
    # ti = Timer()
    # ti.tic()
    net.forward()
    # ti.toc()
    # print ti.total_time
    net.blobs['eltwise_fc1'].data
    return net.blobs['eltwise_fc1'].data

def get_most_similar(feature_to_compare, features_back):
    if len(features_back) == 0:  
        return np.empty((0))  
    similars = []
    for feature in features_back:
        similar = 1 - scipy.spatial.distance.cosine(feature, feature_to_compare)
        similars.append(similar)
    max_similar = max(similars)
    return max_similar, similars.index(max_similar)
  
def read_features(features_txt, features_bin):
    if not os.path.isfile(features_txt):
        print ("features_txt not found!")
        return
    if not os.path.isfile(features_bin):
        print ("features_bin not found!")
        return
    feature_ids = []
    with open(features_txt,'r') as f:
        feature_ids = f.read()
    feature_ids = feature_ids.strip()
    feature_ids = feature_ids.split('\n')
    features = np.fromfile(features_bin, dtype = np.float32)
    features = features.reshape(len(feature_ids),256)
    return features, feature_ids

def get_most_similar_templet(face_img, sex_flag = 'male'):

    face_feature = get_feature(face_img)
    if sex_flag == 'male':
        max_similar, max_similar_index = get_most_similar(face_feature, features_male_template)
        # print max_similar
        male_face_name = features_male_ids[max_similar_index]
        # print ('male: %s' %male_face_name)
        templet_img_align_points = male_points_dic[male_face_name]
        templet_img_align = male_img_dic[male_face_name]
    else:
        max_similar, max_similar_index = get_most_similar(face_feature, features_female_template)
        # print max_similar
        female_face_name = features_female_ids[max_similar_index]
        # print ('female: %s' %female_face_name)
        templet_img_align_points = female_points_dic[female_face_name]
        templet_img_align = female_img_dic[female_face_name]
    
    return templet_img_align, templet_img_align_points

def get_change_baby(baby_name):
    baby_align_img = baby_img_dic[baby_name]
    baby_face_points = baby_points_dic[baby_name]

    return baby_align_img, baby_face_points

if __name__ == '__main__':
    features_female_template, features_female_ids = read_features('lcnn_face/features/face_features_female.txt', 'lcnn_face/features/face_features_female.bin')
    init_model(r"lcnn_face/models/LCNN_deploy.prototxt", r"lcnn_face/models/LCNN_iter_3560000.caffemodel")
    img = cv2.imread('data/test_img/timg-1.jpg')

    feature = get_feature(img)
    max_similar, max_similar_index = get_most_similar(feature, features_female_template)

    print features_female_ids[max_similar_index]
    print max_similar


