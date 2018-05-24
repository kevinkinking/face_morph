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
import signal
import timer

def set_timeout(num, callback):  
    def wrap(func):  
        def handle(signum, frame):
            raise RuntimeError  
        def to_do(*args, **kwargs):
            try:
                signal.signal(signal.SIGALRM, handle)
                signal.alarm(num)
                # print 'start alarm signal.'
                r = func(*args, **kwargs)
                # print 'close alarm signal.'
                signal.alarm(0)
                return r
            except RuntimeError as e:
                return callback()
  
        return to_do
    return wrap

def after_timeout(): 
    # print "do something after timeout."
    return 412, None

def global_init():
    face_match.init_model(config.prototxt, config.caffemodel)
    # print 'face match model success init'
    face_match.init_template_data()
    # print 'template data success load'
    face_process.init_detector()
    # print 'face process model success init'
    global triangle_68_points, triangle_82_points, triangle_78_Oldpoints
    triangle_68_points = init_data.measure_triangle_sample(config.triangle_68)
    triangle_82_points = init_data.measure_triangle_sample(config.triangle_82)
    triangle_78_Oldpoints = init_data.measure_triangle_sample(config.triangle_old_78)
    # print 'triangle data success init'


@set_timeout(config.algo_timeout, after_timeout)
def face_aging_interface_url(img_url, sex_flag='male'):
    try:
        resp = urllib.urlopen(img_url)
        img_data_undecode = np.asarray(bytearray(resp.read()), dtype = 'uint8')
        img = cv2.imdecode(img_data_undecode, cv2.IMREAD_COLOR)
    except Exception as e:
        # print e
        return 401, None

    code, result_imgs, img_aligned = morpher.face_aging_interface(img, triangle_68_points, 
        triangle_82_points, triangle_78_Oldpoints, False, sex_flag)
    # print code
    urls = []
    if code == 202:
        result_imgs = img_util.corp_imgs_scale(result_imgs, 0.7)
        img_aligned = img_util.corp_imgs_scale([img_aligned], 0.7)
        img_aligned_byte = img_util.create_imgs_byte(img_aligned)[0]
        imgs_byte = img_util.create_imgs_byte(result_imgs)
        try:
            url = qiniu_util.qiniu_upload_data(img_aligned_byte, config.img_format)
            urls.append(url)
            for img_byte in imgs_byte:
                url = qiniu_util.qiniu_upload_data(img_byte, config.img_format)
                urls.append(url)
        except Exception as e:
            return 401, None
    return code, urls


if __name__ == '__main__':
    global_init()
    ti =timer.Timer()
    ti.tic()
    code, urls = face_aging_interface_url('data/baby/7.jpg', 'female')
    ti.toc()
    print ti.total_time
    print code
    print urls
