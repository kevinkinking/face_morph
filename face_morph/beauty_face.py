import cv2
import numpy as np

def white_face(img):

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = np.zeros_like(img)

    v1 = 3
    v2 = 1
    dx = v1 * 5 
    fc = v1 * 12.5
    p = -0.5
   
    temp4 = np.zeros_like(img)
    
    temp1 = cv2.bilateralFilter(img,dx,fc,fc)
    temp2 = cv2.subtract(temp1,img);
    #temp2 = cv2.add(temp2,(10,10,10,-100))
    temp3 = cv2.GaussianBlur(temp2,(2*v2 -1,2*v2-1),0.0)
    temp4 = cv2.add(img,temp3)
    dst = cv2.addWeighted(img,p,temp4,1-p,0.0)
    dst = cv2.add(dst,(-5, -5, -5,255))
    
    return dst

if __name__ == '__main__':
    
    img = cv2.imread('11.jpg', cv2.IMREAD_COLOR)
    result_img = white_face(img)
    cv2.imshow('src_img', img)
    cv2.imshow('result_img', result_img)
    cv2.imwrite('111.jpg', result_img)
    cv2.waitKey(-1)
