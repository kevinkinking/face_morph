# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for dlib-based alignment."""

# NOTE: This file has been copied from the openface project.
#  https://github.com/cmusatyalab/openface/blob/master/openface/align_dlib.py

import cv2
import dlib
import numpy as np
import config
import math

TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

INV_TEMPLATE = np.float32([
                            (-0.04099179660567834, -0.008425234314031194, 2.575498465013183),
                            (0.04062510634554352, -0.009678089746831375, -1.2534351452524177),
                            (0.0003666902601348179, 0.01810332406086298, -0.32206331976076663)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)


class AlignDlib:
    """
    Use `dlib's landmark estimation <http://blog.dlib.net/2014/08/real-time-face-pose-estimation.html>`_ to align faces.

    The alignment preprocess faces for input into a neural network.
    Faces are resized to the same size (such as 96x96) and transformed
    to make landmarks (such as the eyes and nose) appear at the same
    location on every image.

    Normalized landmarks:

    .. image:: ../images/dlib-landmark-mean.png
    """

    #: Landmark indices corresponding to the inner eyes and bottom lip.
    INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]

    #: Landmark indices corresponding to the outer eyes and nose.
    OUTER_EYES_AND_NOSE = [36, 45, 33]

    def __init__(self, facePredictor):
        """
        Instantiate an 'AlignDlib' object.

        :param facePredictor: The path to dlib's
        :type facePredictor: str
        """
        assert facePredictor is not None

        #pylint: disable=no-member
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(facePredictor)

    def getAllFaceBoundingBoxes(self, rgbImg):
        """
        Find all face bounding boxes in an image.

        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :return: All face bounding boxes in an image.
        :rtype: dlib.rectangles
        """
        assert rgbImg is not None

        try:
            return self.detector(rgbImg, 1)
        except Exception as e: #pylint: disable=broad-except
            print("Warning: {}".format(e))
            # In rare cases, exceptions are thrown.
            return []

    def getLargestFaceBoundingBox(self, rgbImg, skipMulti=False):
        """
        Find the largest face bounding box in an image.

        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :param skipMulti: Skip image if more than one face detected.
        :type skipMulti: bool
        :return: The largest face bounding box in an image, or None.
        :rtype: dlib.rectangle
        """
        assert rgbImg is not None

        faces = self.getAllFaceBoundingBoxes(rgbImg)
        if (not skipMulti and len(faces) > 0) or len(faces) == 1:
            return max(faces, key=lambda rect: rect.width() * rect.height())
        else:
            return None

    def computer_diatance(self, points):
        left_eyeX = 0
        right_eyex = 0
        left_eyey = 0
        right_eyey = 0
        for i in range(36,42):
            left_eyeX = left_eyeX + points[i][0]
            left_eyey = left_eyey + points[i][1]
        left_eyeX = left_eyeX / 6
        left_eyey = left_eyey / 6

        for i in range(42,48):
            right_eyex = right_eyex + points[i][0]
            right_eyey = right_eyey + points[i][1]
        right_eyex = right_eyex / 6
        right_eyey = right_eyey / 6

        eyes_distance = abs(left_eyeX - right_eyex)
        eyes_centerX = eyes_distance/2 + left_eyeX
        
        distance = abs(points[30][0] - eyes_centerX) 

        return distance

    def modify_landmarks(self, Rotate_M, points):
        align_points=[]
        for i in range(0,len(points)):
            pt=[points[i][0],points[i][1]]
            [[pt[0]], [pt[1]]] = np.dot(Rotate_M, np.array([[pt[0]], [pt[1]], [1]]))
            align_points.append((int(pt[0]),int(pt[1])))
        return align_points

    def findLandmarks(self, rgbImg, bb):
        """
        Find the landmarks of a face.

        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :param bb: Bounding box around the face to find landmarks for.
        :type bb: dlib.rectangle
        :return: Detected landmark locations.
        :rtype: list of (x,y) tuples
        """
        assert rgbImg is not None
        assert bb is not None

        points = self.predictor(rgbImg, bb)
        #return list(map(lambda p: (p.x, p.y), points.parts()))
        return [(p.x, p.y) for p in points.parts()]

    def rotate_img(self, img, center=None, angle=0, crop_size=None, scale=1.0):
        width = img.shape[1]
        height = img.shape[0]
        if center is None:
            center = (width / 2, height / 2)
        # print center
        rot_m = cv2.getRotationMatrix2D(center, angle, scale)
        if crop_size is not None:
            rot_m[0][2] += crop_size[0] / 2.0 - center[0]
            rot_m[1][2] += crop_size[1] / 2.0 - center[1]
        img = cv2.warpAffine(img, rot_m, crop_size)
        return img

    def rotate_points(self, points, center=None, angle=0, crop_size=None, scale=1.0):
        if center is None:
            center = (0, 0)

        rot_m = cv2.getRotationMatrix2D(center, angle, scale)
        if crop_size is not None:
            rot_m[0][2] += crop_size[0] / 2.0 - center[0]
            rot_m[1][2] += crop_size[1] / 2.0 - center[1]
        points = self.modify_landmarks(rot_m, points)
        # rotated_points = []
        # for point in points:
        #     rotated_points.append([point[0], point[1]])
        return points

    def caculate_vertical_angle(self, point_left_eye, point_right_eye, point_bottom_lip):
        x0 = point_left_eye[0]
        y0 = point_left_eye[1]
        x1 = point_right_eye[0]
        y1 = point_right_eye[1]
        a = y1 - y0
        b = x0 - x1
        c = x1 * y0 - x0 * y1
        m = point_bottom_lip[0]
        n = point_bottom_lip[1]
        point_vv = ((b*b*m-a*b*n-a*c)/(a*a+b*b),(a*a*n-a*b*m-b*c)/(a*a+b*b))
        x_dis = point_vv[0] - point_bottom_lip[0]
        y_dis = abs(point_vv[1] - point_bottom_lip[1])

        angle = math.atan(float(x_dis) / y_dis) / 3.1415926 * 180
        return angle

    def align_stable(self, rgbImg, min_value=1.2, max_value=5, bb=None,
              landmarks=None, skipMulti=False, scale=0.5):
        if rgbImg is None:
            return 401, None, None, None
        if len(rgbImg.shape) < 3:
            return 404, None, None, None
        if rgbImg.shape[2] == 4:
            rgbImg=cv2.cvtColor(rgbImg,cv2.COLOR_BGRA2BGR) 
        rgbImg_height = rgbImg.shape[0]
        rgbImg_width = rgbImg.shape[1]
        if rgbImg_width < config.min_size[0] or rgbImg_height < config.min_size[1]:
            return 402, None, None, None
        rate = float(rgbImg_height) / rgbImg_width
        if rate > 2.5 or rate < 0.4:
            return 403, None, None, None

        new_rgbImg_width = 0
        new_rgbImg_height = 0

        if rgbImg_width > rgbImg_height:
            new_rgbImg_height = config.detect_size[1]
            new_rgbImg_width = int((float(new_rgbImg_height) / rgbImg_height) * rgbImg_width)
        else:
            new_rgbImg_width = config.detect_size[0]
            new_rgbImg_height = int((float(new_rgbImg_width) / rgbImg_width) * rgbImg_height)
        rgbImg = cv2.resize(rgbImg, (new_rgbImg_width, new_rgbImg_height))

        if bb is None:
            bb = self.getLargestFaceBoundingBox(rgbImg, skipMulti)
            if bb is None:
                print 'no face detect'
                return 405, None, None, None

        face_width =  bb.right() - bb.left()
        face_height = bb.bottom() - bb.top()
        
        scale_m = config.crop_size[0] * scale / face_width

        if face_width * min_value > rgbImg.shape[1] or face_width * max_value < rgbImg.shape[1]:
            print 'face is too small or big'
            return 406, None, None, None
        if landmarks is None:
            landmarks = self.findLandmarks(rgbImg, bb)

        point_left_eye = landmarks[39]
        point_right_eye = landmarks[42]
        point_bottom_lip = landmarks[57]
        point_nose = landmarks[30]
        angle = self.caculate_vertical_angle(point_left_eye, point_right_eye, point_bottom_lip)
        # print angle
        img_aligned = self.rotate_img(rgbImg, center=point_nose, angle= angle, crop_size=config.crop_size, scale=scale_m)
        landmarks_aligned = self.rotate_points(landmarks, center=point_nose, angle= angle, crop_size=config.crop_size, scale=scale_m)
        
        face_distance = self.computer_diatance(landmarks_aligned)
        if face_distance > config.max_noseEye_distance:
            print ('face angle=%d > %d. is too big' %(face_distance , config.max_noseEye_distance))
            return 407, None, None, None

        face_width_aligned_half = face_width * scale_m / 2
        face_height_aligned_half = face_height * scale_m / 2
        face_left = int(config.crop_size[0] / 2 - face_width_aligned_half)
        face_right = int(config.crop_size[0] / 2 + face_width_aligned_half)
        face_top = int(config.crop_size[1] / 2 - face_height_aligned_half)
        face_bottom = int(config.crop_size[1] / 2 + face_height_aligned_half)

        face_img_aligned = img_aligned[face_top:face_bottom, face_left:face_right]
        face_img_aligned = cv2.resize(face_img_aligned, (128, 128))
        return 201, img_aligned, face_img_aligned, landmarks_aligned


    #pylint: disable=dangerous-default-value
    def align(self, rgbImg, min_value=1.2, max_value=5, bb=None,
              landmarks=None, landmarkIndices=INNER_EYES_AND_BOTTOM_LIP,
              skipMulti=False, scale=0.3):
        img_dim = 800
        if rgbImg is None:
            return 401, None, None, None
        if len(rgbImg.shape) < 3:
            return 404, None, None, None
        if rgbImg.shape[2] == 4:
            rgbImg=cv2.cvtColor(rgbImg,cv2.COLOR_BGRA2BGR) 
        rgbImg_height = rgbImg.shape[0]
        rgbImg_width = rgbImg.shape[1]
        if rgbImg_width < 300 or rgbImg_height < 300:
            return 402, None, None, None
        rate = float(rgbImg_height) / rgbImg_width
        if rate > 2.5 or rate < 0.4:
            return 403, None, None, None

        new_rgbImg_width = 0
        new_rgbImg_height = 0

        if rgbImg_width > rgbImg_height:
            new_rgbImg_height = 900
            new_rgbImg_width = int((float(new_rgbImg_height) / rgbImg_height) * rgbImg_width)
        else:
            new_rgbImg_width = 800
            new_rgbImg_height = int((float(new_rgbImg_width) / rgbImg_width) * rgbImg_height)
        rgbImg = cv2.resize(rgbImg, (new_rgbImg_width, new_rgbImg_height))
        if bb is None:
            bb = self.getLargestFaceBoundingBox(rgbImg, skipMulti)
            if bb is None:
                print 'no face detect'
                return 405, None, None, None

        face_width =  bb.right() - bb.left()

        if face_width * min_value > rgbImg.shape[1] or face_width * max_value < rgbImg.shape[1]:
            print 'face is too small or big'
            return 406, None, None, None
        if landmarks is None:
            landmarks = self.findLandmarks(rgbImg, bb)

        npLandmarks = np.float32(landmarks)
        npLandmarkIndices = np.array(landmarkIndices)

        H = cv2.getAffineTransform(npLandmarks[npLandmarkIndices],
                                   img_dim * MINMAX_TEMPLATE[npLandmarkIndices] * scale + img_dim * (1 - scale) / 2)

        face_landmarks_aligned = self.modify_landmarks(H, npLandmarks)

        face_distance = self.computer_diatance(face_landmarks_aligned)
        if face_distance > config.max_noseEye_distance:
            print ('face angle=%d > %d. is too big' %(face_distance , config.max_noseEye_distance))
            return 407, None, None, None

        face_landmarks_aligned = [(point[0] - 100, point[1])for point in face_landmarks_aligned]

        img_aligned = cv2.warpAffine(rgbImg, H, (img_dim, img_dim))
        
        img_aligned = img_aligned[0:800,100:700]

        face_img_dim = 128
        face_scale = 1.0
        face_H = cv2.getAffineTransform(npLandmarks[npLandmarkIndices],
                                   face_img_dim * MINMAX_TEMPLATE[npLandmarkIndices] * face_scale + face_img_dim * (1 - face_scale) / 2)

        face_img_aligned = cv2.warpAffine(rgbImg, face_H, (face_img_dim, face_img_dim))
        
        return 201, img_aligned, face_img_aligned, face_landmarks_aligned

def crop_dataset(img_dir, output_dir, scale = 1):
    import os
    face_align = AlignDlib('face_util/shape_predictor_68_face_landmarks.dat')
    img_names = os.listdir(img_dir)
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        output_img_path = os.path.join(output_dir, img_name)
        img = cv2.imread(img_path)
        rect = face_align.getLargestFaceBoundingBox(img)
        if rect is None:
            continue
        width = rect.right() - rect.left()
        height = rect.bottom() - rect.top() 


        new_left = rect.left() - int(width * scale)
        if new_left < 0:
            new_left = 0
        new_top = rect.top() - int(height * scale)
        if new_top < 0:
            new_top = 0
        new_right = rect.right() + int(width * scale)
        if new_right > img.shape[1]:
            new_right = img.shape[1]
        new_bottom = rect.bottom() + int(height * scale)
        if new_bottom > img.shape[0]:
            new_bottom >= img.shape[0]
        crop_img = img[new_top:new_bottom, new_left:new_right]
        print img_name
        cv2.imwrite(output_img_path, crop_img)

def show_dataset(img_dir):
    import os
    face_align = AlignDlib('face_util/shape_predictor_68_face_landmarks.dat')
    img_names = os.listdir(img_dir)
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        code, img, img_face, face_landmarks_aligned = face_align.align(rgbImg=img,min_value=1.2, max_value=15)
        if face_landmarks_aligned is None:
            print img_name + 'is none'
            continue
        for i in range(0, len(face_landmarks_aligned)):
            cv2.circle(img, face_landmarks_aligned[i], 7, (0, 255, 0))
        print img_name
        cv2.imshow('show', img)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()



if __name__ == '__main__':

    face_align = AlignDlib('face_util/shape_predictor_68_face_landmarks.dat')
    
    img = cv2.imread('./data/3131860710_SXXL.jpg')
    code, img_aligned, face_img_aligned, landmarks_aligned= face_align.align_stable(img)
    print code
    if code == 201:
        for i in range(0, len(landmarks_aligned)):
            cv2.circle(img_aligned, landmarks_aligned[i], 7, (0, 255, 0))   

        cv2.imshow('show', img_aligned)
        cv2.imshow('show1', face_img_aligned)
        cv2.waitKey(0)

    # face_align = AlignDlib('face_util/shape_predictor_68_face_landmarks.dat')
    # img = cv2.imread('./face_util/test.jpg')
    # points = [(200, 200), (600, 100), (400, 400)]
    # center = (400, 400)
    
    # angle = face_align.caculate_vertical_angle(points[0], points[1], points[2])
    # img = face_align.rotate_img(img, center=center, angle= -angle)
    
    # print angle

    # points = face_align.rotate_points(points, center=center, angle= -angle)
    # print points
    # for i in range(0, len(points)):
    #     cv2.circle(img, points[i], 7, (0, 255, 0))
    # cv2.imshow('show', img)
    # cv2.waitKey(0)




    # crop_dataset('/home/bbt/male_0523', '/home/bbt/male_0523')
    # show_dataset('/home/bbt/work/face_morph_master/data/template_img_crop_big/male')
    # img = cv2.imread('data/template_img/female/VCG21gic5474042.jpg', cv2.IMREAD_COLOR)

    # face_align = AlignDlib('face_util/shape_predictor_68_face_landmarks.dat')
    
    # code, img, img_face, face_landmarks_aligned1 = face_align.align(rgbImg=img,min_value=1.2, max_value=8)
    # print face_landmarks_aligned1
    # for i in range(0, len(face_landmarks_aligned1)):
    #     cv2.circle(img, face_landmarks_aligned1[i], 7, (0, 255, 0))
    # #cv2.rectangle(img,(rectangle_rect[0],rectangle_rect[1]),(rectangle_rect[3],rectangle_rect[2]), (255,0,0))
    # cv2.imshow('show', img)
    # cv2.waitKey(0)

