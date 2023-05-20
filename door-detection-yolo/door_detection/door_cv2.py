import numpy as np
import cv2
import os
from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries

from core.settings import Door_Cv2_CONFIG

#parameter
scale = Door_Cv2_CONFIG.scale
sigma = Door_Cv2_CONFIG.sigma
min_size = Door_Cv2_CONFIG.min_size

def detection_cv2(img_path:str, x:int, y:int, w:int, h:int):
        

    #reading images
    pic=cv2.imread(img_path) 
    
    #increase boundary to find door
    x1 = int(0.5*x)
    y1 = int(0.8*y)
    w1 = int(1.5*w)
    h1 = int(1.2*h)

    img = pic[y1:y+h1,x1:x+w1,:]

    #felzenberg algorithm
    segments_fz = felzenszwalb(img, scale=scale, sigma=sigma, min_size=min_size)

    min_size_s = np.shape(img)[0] * np.shape(img)[1] - len(np.where(segments_fz)[0])

    segments_fz = felzenszwalb(img, scale=150, sigma=0.5, min_size=min_size_s)


    s=np.copy(segments_fz)
    #making a boundry
    img_out_path = os.path.join(os.path.dirname(img_path),"out.jpg")
    cv2.imwrite(img_out_path,np.uint8(255*mark_boundaries(img, s)))
    # pic[y1:y+h1,x1:x+w1,:] = np.uint8(255*mark_boundaries(img, segments_fz))
    # plt.imshow(np.uint8(255*mark_boundaries(img, segments_fz)))
                
    return img_out_path