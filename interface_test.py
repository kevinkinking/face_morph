from face_morph import morpher
import cv2
import config
from lcnn_face import face_match, init_data
from face_util import face_process
from timer import Timer
import urllib
import numpy as np
from media_util import img_util
from media_util import qiniu_util


def global_init():
    face_match.init_model(config.prototxt, config.caffemodel)
    print 'face match model success init'
    face_match.init_template_data()
    print 'template data success load'
    face_process.init_detector()
    print 'face process model success init'
    global triangle_68_points, triangle_82_points, triangle_78_Oldpoints
    triangle_68_points = init_data.measure_triangle_sample(config.triangle_68)
    triangle_82_points = init_data.measure_triangle_sample(config.triangle_82)
    triangle_78_Oldpoints = init_data.measure_triangle_sample(config.triangle_old_78)
    print 'triangle data success init'
def face_aging_interface_url(img_url, sex_flag):
    try:
        resp = urllib.urlopen(img_url)
        img_data_undecode = np.asarray(bytearray(resp.read()), dtype = 'uint8')
        img = cv2.imdecode(img_data_undecode, cv2.IMREAD_COLOR)
    except Exception as e:
        print e
        return 401, None, None
    ti = Timer()
    ti.tic()
    code, result_imgs, img_aligned = morpher.face_aging_interface(img, triangle_68_points, triangle_82_points, triangle_78_Oldpoints, False, 'female')
    ti.toc()
    print ti.total_time
    result_imgs = img_util.corp_imgs_scale(result_imgs, 0.7)
    img_aligned = img_util.corp_imgs_scale([img_aligned], 0.7)[0]
    gif_byte = img_util.create_gif_byte(result_imgs)
    imgs_byte = img_util.create_imgs_byte(result_imgs)
    urls = []
    ti = Timer()
    ti.tic()
    url = qiniu_util.qiniu_upload_data(gif_byte, '.gif')
    urls.append(url)
    for img_byte in imgs_byte:
        url = qiniu_util.qiniu_upload_data(img_byte, config.img_format)
        urls.append(url)
    ti.toc()
    print ti.total_time
    print urls
    return code, urls, result_imgs, img_aligned


if __name__ == '__main__':
    global_init()
    code, urls, result_imgs, img_aligned = face_aging_interface_url('data/baby/21.jpg', 1)
    print code
    if code == 202:
        for i in range(len(result_imgs)):
            img_iter = result_imgs[i]
            cv2.imwrite('data/'+str(i)+'.jpg',img_iter)
            # img_new = np.zeros((560,840,3),dtype = 'uint8')
            # img_new[:,0:420,:] = img_aligned
            # img_new[:,420:840,:] = img_iter
            # cv2.imshow('show', img_new)
            # cv2.waitKey(500)

# if __name__ == '__main__':
#     global_init()
#     code, urls, result_imgs, img_aligned, result_img_o = face_aging_interface_url('data/baby/37.jpg', 1)
#     print code

#     if code == 202:
#         for i in range(len(result_imgs)):
#             img_iter = result_imgs[i]
#             img_iter_old = result_img_o[i]
#             (h,w,c) = img_aligned.shape
#             save_img = np.zeros((h,3*w,c), dtype = img_aligned.dtype)
#             save_img[:,0:w,:] = img_aligned
#             save_img[:,w:2*w,:] = img_iter
#             save_img[:,2*w:3*w,:] = img_iter_old
#             cv2.imwrite('data/'+str(i)+'.jpg',save_img)