import align_dlib as align
import numpy as np
import config
import cv2
import dlib
import os

def init_detector():
    global face_detector
    face_detector = align.AlignDlib(config.shape_predictor_68)
    return face_detector

def measure_triangle_sample(file_name):

    triangle = []
    with open(file_name) as file :
        for line in file :
            x,y,z = line.split()
            triangle.append((int(x), int(y), int(z)))
    return triangle

def get_face_rectangle(points, img):
    
    face_rectangle = []
    (h,w,c) = img.shape
    top = min([p[0] for p in points])
    left = min([p[1] for p in points])
    bottom = max([p[0] for p in points])
    right = max([p[1] for p in points])
    
    print top,left,bottom,right

    height = abs(bottom - top)
    width = abs(right - left)

    top = int(max(0, top - height))
    left = int(max(0, left - width))
    bottom = int(min(h, bottom + height))
    right = int(min(w, right + width))
    
    face_rectangle = [top, left, bottom, right]

    return face_rectangle
    
def computer_angle(points):
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

    x = np.array([right_eyex - left_eyeX , right_eyey - left_eyey])
    y = np.array([right_eyex - left_eyeX , 0])

    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    if Lx == 0.0:
        Lx = 1
    if Ly == 0.0:
        Ly = 1
    cos_angle = x.dot(y) / (Lx * Ly)

    angle = np.arccos(cos_angle)

    angle = abs(angle * 360 / 2 / np.pi)

    return angle

def computer_diatance(points):
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

def get_landmarks(img):
    points = []
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # cv2.equalizeHist(gray)
    faces = face_detector.getAllFaceBoundingBoxes(gray)

    if len(faces) == 0:
        print 'no face detect'
        return 405, None
    face = max(faces, key=lambda rect: rect.width() * rect.height())

    face = dlib.rectangle(face.left(), face.top(), face.right(),
                          face.bottom())
    bb = face
    w = face.right() - face.left()
    h = face.bottom() - face.top()
    
    points = face_detector.findLandmarks(img, bb)

    return 201, points

def show_landmarks(img, points):
    
    for i in range(0, len(points)):
        cv2.circle(img, points[i], 7, (0, 255, 0))
    #cv2.rectangle(img, (face.left(),face.top()), (face.right(),face.bottom()), (255,0,0))
    cv2.imshow('show', img)
    cv2.waitKey(0)

def show_align_points(img, min_value=1, max_value=5):

    img_align, face_img_aligned, points = face_detector.align(rgbImg = img, min_value=min_value, max_value=max_value)
    for i in range(0, len(points)):
        cv2.circle(img_align, points[i], 3, (0, 255, 0))
    cv2.imshow('output', img_align)
    cv2.waitKey(-1)

def crop_faces_dir(img_dir,faces_crop_dir, min_value=1, max_value=5):

    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir,img_name)
        print img_path
        img = cv2.imread(img_path)
        _, _, face_img_aligned, _ = face_detector.align(rgbImg = img, min_value=min_value, max_value=max_value)
        if face_img_aligned is not None:
            face_img_path = os.path.join(faces_crop_dir,img_name)
            cv2.imwrite(face_img_path, face_img_aligned)

if __name__ == '__main__':

    img = cv2.imread('face_util/test/7.jpg',cv2.IMREAD_COLOR)
    detector = init_detector()
    tri = measure_triangle_sample('face_morph/face_features_malewww.txt')
    show_align_points(img, detector, tri)
    #crop_faces_dir('/home/babytree/face_morph_master_20180506/data/template_img/male',
    #    '/home/babytree/face_morph_master_20180506/data/template_img_crop/male')
