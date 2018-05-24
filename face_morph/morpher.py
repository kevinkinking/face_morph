import cv2
import numpy as np
import time
import os
import config

import face_morph
from face_util import face_process
from lcnn_face import timer,face_match
from timer import Timer

#transform dst_img points according src_img
def transformation_points(src_img, src_points, dst_img, dst_points):
    src_points = src_points.astype(np.float64)
    dst_points = dst_points.astype(np.float64)
    
    c1 = np.mean(src_points, axis=0)
    c2 = np.mean(dst_points, axis=0)

    src_points -= c1
    dst_points -= c2

    s1 = np.std(src_points)
    if s1 == 0.0:
        s1 = 1.0

    s2 = np.std(dst_points)
    if s2 == 0.0:
        s2 = 1.0

    src_points /= s1
    dst_points /= s2

    u, s, vt = np.linalg.svd(src_points.T * dst_points)
    r = (u * vt).T

    m = np.vstack([np.hstack(((s2 / s1) * r, c2.T - (s2 / s1) * r * c1.T)), np.matrix([0., 0., 1.])])

    output = cv2.warpAffine(dst_img, m[:2],
                            (src_img.shape[1], src_img.shape[0]),
                            borderMode=cv2.BORDER_TRANSPARENT,
                            flags=cv2.WARP_INVERSE_MAP)
    return output


def tran_matrix(src_img, src_points, dst_img, dst_points):
    h = cv2.findHomography(dst_points, src_points)
    output = cv2.warpAffine(dst_img, h[0][:2], (src_img.shape[1], src_img.shape[0]),
                            borderMode=cv2.BORDER_TRANSPARENT,
                            flags=cv2.WARP_INVERSE_MAP)

    return output


def correct_color(img1, img2, landmark):
    blur_amount = 0.4 * np.linalg.norm(
        np.mean(landmark[face_morph.LEFT_EYE_POINTS], axis=0)
        - np.mean(landmark[face_morph.RIGHT_EYE_POINTS], axis=0)
    )
    blur_amount = int(blur_amount)

    if blur_amount % 2 == 0:
        blur_amount += 1

    img1_blur = cv2.GaussianBlur(img1, (blur_amount, blur_amount), 0)
    img2_blur = cv2.GaussianBlur(img2, (blur_amount, blur_amount), 0)

    img2_blur += (128 * (img2_blur <= 1.0)).astype(img2_blur.dtype)

    return img2.astype(np.float64) * img1_blur.astype(np.float64) / img2_blur.astype(np.float64)

#transform src_img src_points according dst_points in face_area
def tran_src(src_img, src_points, dst_points, dt, face_area=None):
    jaw = face_morph.JAW_END

    dst_list = dst_points \
               + face_morph.matrix_rectangle1(face_area[0], face_area[1], face_area[2], face_area[3]) \
               + face_morph.matrix_rectangle(0, 0, src_img.shape[1], src_img.shape[0])

    src_list = src_points \
               + face_morph.matrix_rectangle1(face_area[0], face_area[1], face_area[2], face_area[3]) \
               + face_morph.matrix_rectangle(0, 0, src_img.shape[1], src_img.shape[0])

    res_img = np.zeros(src_img.shape, dtype=src_img.dtype)

    for i in range(0, len(dt)):
        t_src = []
        t_dst = []

        for j in range(0, 3):
            t_src.append(src_list[dt[i][j]])
            t_dst.append(dst_list[dt[i][j]])

        face_morph.affine_triangle(src_img, res_img, t_src, t_dst)

    return res_img

#merge dst_img into src_img
def merge_img(src_img, dst_img, dst_matrix, dst_points, k_size=None, mat_multiple=None):

    face_mask = np.zeros(src_img.shape, dtype=src_img.dtype)

    for group in face_morph.OVERLAY_POINTS:
        cv2.fillConvexPoly(face_mask, cv2.convexHull(dst_matrix[group]), (255, 255, 255))

    center_x=0
    center_y=0
    for i in range(0,len(dst_points)):
        center_x=center_x+dst_points[i][0]
        center_y=center_y+dst_points[i][1]
    center_x=int(center_x/len(dst_points))
    center_y=int(center_y/len(dst_points))
    center = (center_x,center_y)

    if mat_multiple:
        mat = cv2.getRotationMatrix2D(center, 0, mat_multiple)
        face_mask = cv2.warpAffine(face_mask, mat, (face_mask.shape[1], face_mask.shape[0]))
    
    if k_size:
        face_mask = cv2.blur(face_mask, k_size, center)
    
    img_merge = cv2.seamlessClone(np.uint8(dst_img), src_img, face_mask, center, cv2.NORMAL_CLONE)
    
    return img_merge

#merge dst_img into src_img according alpha, if alpha is 0, return src_img, and so on
def morph_img(src_img, src_points, dst_img, dst_points, dt, alpha=0.5):

    morph_points = []

    src_img = src_img.astype(np.float32)
    dst_img = dst_img.astype(np.float32)

    #res_img = np.zeros(src_img.shape, src_img.dtype)
    res_img = np.copy(src_img)
    for i in range(0, len(src_points)):
        x = (1 - alpha) * src_points[i][0] + alpha * dst_points[i][0]
        y = (1 - alpha) * src_points[i][1] + alpha * dst_points[i][1]
        morph_points.append((x, y))

    for i in range(0, len(dt)):
        t1 = []
        t2 = []
        t = []

        for j in range(0, 3):
            t1.append(src_points[dt[i][j]])
            t2.append(dst_points[dt[i][j]])
            t.append(morph_points[dt[i][j]])

        face_morph.morph_triangle(src_img, dst_img, res_img, t1, t2, t, alpha)

    return res_img, morph_points

def get_matrix_and_box(points):
    matrix_data = []
    points_data = []
    for i in range(0,len(points)):
        a = int(points[i][0])
        b = int(points[i][1])
        matrix_data.append([a,b])
        points_data.append((a,b))
    matrix_data = np.matrix(matrix_data)

    return matrix_data,points_data

def center_point(points):
    center_x = 0
    center_y = 0
    for i in range(48,68):
        center_x = center_x + points[i][0]
        center_y = center_y + points[i][0]
    center = (int(center_x/20),int(center_y/20))

    return center

def face_transform(img_aligned, img_matrix, templet_img_align, templet_matrix):

    img_trans_aligned = transformation_points(src_img=templet_img_align, src_points=templet_matrix[face_morph.FACE_POINTS],
                                    dst_img=img_aligned, dst_points=img_matrix[face_morph.FACE_POINTS])

    code, img_trans_aligned_points = face_process.get_landmarks(img_trans_aligned)
    if code != 201:
        return code, None, None, None

    trans_matrix, trans_points = get_matrix_and_box(img_trans_aligned_points)
    return code, img_trans_aligned, trans_matrix, trans_points

def face_morph_merge(templet_img_align, templet_points, img_trans_aligned, trans_points, triangle_68_points, triangle_82_points, alpha):
    
    merge_two_img, morph_points_two = morph_img(templet_img_align, templet_points, img_trans_aligned, trans_points, triangle_68_points, alpha)
    
    morp_matrix,morp_points = get_matrix_and_box(morph_points_two)
    
    trans_templet_img = tran_src(templet_img_align, templet_points, morp_points, triangle_82_points, config.face_area)
    
    result_img = merge_img(trans_templet_img, merge_two_img, morp_matrix, morp_points, config.k_size, config.mat_multiple)

    return result_img

def face_morge_old(templet_img_align, templet_points, img_trans_aligned, trans_points, triangle_78_Oldpoints, alpha):

    templet_points.extend(config.append_points)
    trans_points.extend(config.append_points)

    imgMorph, _ = morph_img(templet_img_align, templet_points, img_trans_aligned, trans_points, triangle_78_Oldpoints, alpha)

    return imgMorph

def face_aging_interface(img,
        triangle_68_points,
        triangle_82_points,
        triangle_78_Oldpoints,
        version_flag=True,
        sex_flag='male'):

    code, img_aligned, face_img_aligned , img_aligned_points = face_process.face_detector.align_stable(img)
    if code != 201:
        return code, None, None

    img_matrix, img_points = get_matrix_and_box(img_aligned_points)

    templet_img_align, templet_img_align_points = face_match.get_most_similar_templet(face_img_aligned, sex_flag)
    
    templet_matrix,templet_points = get_matrix_and_box(templet_img_align_points)
    
    
    if version_flag == True:
        code, img_trans_aligned, trans_matrix, trans_points = face_transform(img_aligned, img_matrix, templet_img_align, templet_matrix)
        if img_trans_aligned is None or trans_matrix is None or trans_points is None:
            return code, None, None

        result_imgs = []
        for alpha in config.alphas:
            result_img = face_morph_merge(templet_img_align, templet_points, img_trans_aligned, trans_points, triangle_68_points, triangle_82_points, alpha)
            result_imgs.append(result_img)
        return 202, result_imgs, img_aligned
    else:
        result_imgs = []
        for alpha in config.alphas:
            result_img = face_morge_old(templet_img_align, templet_points, img_aligned, img_points, triangle_78_Oldpoints, alpha)
            result_imgs.append(result_img)
        return 202, result_imgs, img_aligned