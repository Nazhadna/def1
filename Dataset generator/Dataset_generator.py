import numpy as np
import cv2
from PIL import Image
import sys

'''
This script makes dataset generation a bit easier.

'''

if __name__ == '__main__':
    canvas = cv2.imread(sys.argv[1])
    bb,gg,rr = cv2.split(canvas)
    size = len(sys.argv)-1
    mask = np.zeros(rr.shape,np.uint8)
    for defect in sys.argv[2:size]:
        defect_poly = np.load('polygons\\'+defect+'.npy')
        defect_type,defect_loc,defect_num = defect.split('-')
        defect_source = cv2.imread('defects'+defect_loc+'.jpg')
        b,g,r = cv2.split(defect_source)
        mask_inter = np.zeros(rr.shape,np.uint8)
        cv2.fillPoly(mask,defect_poly,int(defect_type))
        cv2.fillPoly(mask_inter,defect_poly,255)
        deb = Image.fromarray(mask_inter)
        negative_inter = cv2.bitwise_not(mask_inter)

        rr = cv2.bitwise_and(rr,negative_inter)
        gg = cv2.bitwise_and(gg,negative_inter)
        bb = cv2.bitwise_and(bb,negative_inter)
        
        r = cv2.bitwise_and(r,mask_inter)
        g = cv2.bitwise_and(g,mask_inter)
        b = cv2.bitwise_and(b,mask_inter)
        
        rr += r
        gg += g
        bb += b
    
    new_sample = cv2.merge([rr,gg,bb])
    new_sample = Image.fromarray(new_sample)
    new_sample.save('images\\'+sys.argv[-1]+'.png')
    np.save('labels\\'+sys.argv[-1]+'.npy',mask)