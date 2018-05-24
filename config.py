import os
import sys

#filter face
max_face_angle = 4
max_noseEye_distance = 15

#transform face area
face_area=[140, 50, 330, 550]
# face_area=[50, 30, 500, 485]

k_size = (1, 1)

min_size = (300, 300)
detect_size = (800, 1000)
crop_size = (600, 800) #must < detect_size
mat_multiple = 0.90
# alphas = [0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]
alphas = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30,0.20]
# alphas = [0.80, 0.6, 0.4, 0.3]
gif_indexs = [0, 2, 5, 7]

append_points = [(0,650), (599,650), (0,0), (0,400),(0,799), (300,799), (599,799), (599,400),(599,0), (300,0)]

#
shape_predictor_68 = 'face_util/shape_predictor_68_face_landmarks.dat'

#
triangle_82 = 'face_morph/tri_82.txt'
triangle_68 = 'face_morph/tri_64.txt'
triangle_old_78 = 'face_morph/tri_old.txt'

#crop face
template_ManImg_path = r'data/template_img/male/'
save_ManImg_path = r'data/template_img_crop/male/'
template_feManImg_path = 'data/template_img/female/'
save_feManImg_path = r'data/template_img_crop/female/'

#get face feature
prototxt = r"lcnn_face/models/LCNN_deploy.prototxt"
caffemodel = r"lcnn_face/models/LCNN_iter_3560000.caffemodel"
save_male_txt = r'lcnn_face/features/face_features_male.txt'
save_female_txt = r'lcnn_face/features/face_features_female.txt'
save_male_feature = r'lcnn_face/features/face_features_male.bin'
save_female_feature = r'lcnn_face/features/face_features_female.bin'

#save align template images and keypoints file
save_AlignMaleImg_dir = r'data/template_img_align/male/images/'
save_AlignMalePoint_dir = r'data/template_img_align/male/keypoints/'
save_AlignFemaleImg_dir = r'data/template_img_align/female/images/'
save_AlignFemalePoint_dir = r'data/template_img_align/female/keypoints/'

img_format = '.jpg'
img_format_values = ['.jpg', '.png']

gif_speed = 0.2

algo_timeout = 5