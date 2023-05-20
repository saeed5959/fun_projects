#!/usr/bin/env python
# coding: utf-8

# In[2]:


#doing float instead of int
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix,csr_matrix
from scipy.sparse.linalg import spsolve

#making a mask
# """"
# def construct_polygon(event,x,y,flags,points):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print (y, " " , x)
#         points.append((x,y))
#     return

# background_img = cv2.imread('car.jpg')
# border_img = cv2.imread('background.jpg')
# border_img = border_img.astype(np.float)
# m,n = len(background_img),len(background_img[0])
# points = []
# cv2.imshow('get_input',background_img)
# cv2.setMouseCallback('get_input',construct_polygon,points)
# cv2.waitKey(80000)
# cv2.destroyWindow('get_input')

# mask = np.zeros((m,n),np.uint8)
# cv2.fillPoly(mask, np.array([points]), (255,255,255))
# ret, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
# cv2.imshow('mask', mask)
# cv2.waitKey(80000)
# cv2.destroyWindow('mask')
# """"

#reading images
mask_s=cv2.imread('mask.jpg',0)
s=cv2.imread('car.jpg')
t=cv2.imread('background.jpg')

#croping image that must replace to a foreground image
t_ = t[100:100+435,200:200+538,:]
#unknown point
v = np.zeros((435,538,3),np.float)

s = s.astype(np.float)
t_ = t_.astype(np.float)

#nonzero points in mask
white_point = np.nonzero(mask_s)

#coordinate of nonzero points in mask
coor = np.zeros((435,538))

#number of nonzero points
num_white = len(white_point[0])

k=0
#making coordinate of nonzero points in mask
for i in range(0,num_white):
    x = white_point[1][i]
    y = white_point[0][i]
    coor[y,x] = k
    k = k+1
    
#gradian 
g = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]],np.float) 
#applying gradian to foreground image
gradian = cv2.filter2D(s,-1,g)

#making a sparse matrix and Ax=B
A = lil_matrix((num_white,num_white))
B = np.zeros((num_white,3),np.float)

coor = np.uint32(coor)
#making A and B for any nonzero point
for i in range(0,435):
    for j in range(0,538):
        
        #for points out of mask , the point must be choosen from background image 
        if mask_s[i,j]==0:
            v[i,j,:] = t_[i,j,:]
        
        else:
            #for points inside of mask , if neighbour of point is inside of mask , construct A 
            if mask_s[i-1,j]==255:
                A[coor[i,j],coor[i-1,j]] = -1
            
            #for points inside of mask , if neighbour of point is outside of mask , construct B
            else:
                for k in range(0,3):
                    B[coor[i,j],k] +=  t_[i-1,j,k]
                
            if mask_s[i+1,j]==255:
                A[coor[i,j],coor[i+1,j]] = -1
                
            else:
                 for k in range(0,3):
                    B[coor[i,j],k] +=  t_[i+1,j,k]
                
            if mask_s[i,j-1]==255:
                A[coor[i,j],coor[i,j-1]] = -1
                
            else:
                 for k in range(0,3):
                    B[coor[i,j],k] +=  t_[i,j-1,k]
                
            if mask_s[i,j+1]==255:
                A[coor[i,j],coor[i,j+1]] = -1
                
            else:
                 for k in range(0,3):
                    B[coor[i,j],k] +=  t_[i,j+1,k]
            
            #replacing the value in indice of point in A equal 4
            A[coor[i,j],coor[i,j]] = 4
            #replacing the value in indice of point in B equal the gradian
            for k in range(0,3):
                    B[coor[i,j],k] +=  gradian[i,j,k]

#solving Ax=B 
A_ = A.tocsr()
solve = spsolve(A_,B)

#returning value from solve to the contributed point
for i in range(0,num_white):
    x = white_point[1][i]
    y = white_point[0][i]
    for k in range(0,3):
        v[y,x,k] = solve[i,k]
        
#making a limit for v   
v[v>255]=255
v[v<0]=0

#replace and svae image
v = v.astype(np.uint8)
t[100:100+435,200:200+538,:] = v
cv2.imwrite('magic.jpg',t)

