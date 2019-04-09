#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tqdm
import os
from tqdm import tqdm 
from PIL import Image
import cv2


# In[ ]:


def getFlow1(img1, img2):
    flow = cv2.calcOpticalFlowFarneback(img2, img1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    ang =  0.5 * ang * 180 / np.pi
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return ang, mag


def getFlow2(img1, img2):
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p0 = cv2.goodFeaturesToTrack(img1, mask = None, **feature_params)
    p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, **lk_params)
    color = np.random.randint(0,255,(100,3))
    mask = np.zeros_like(img1)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)

    return mask

def getCanny(img, t1, t2):
    return cv2.Canny(img, t1, t2)



def getFourier(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    return magnitude_spectrum


def getJET(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow,ccol = rows/2 , cols/2
    crow = int(crow)
    ccol = int(ccol)
    fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back


# In[ ]:


root = root = "D:\\Academics\\Sem6\\Deep Learning\\DL_A2\\new\\"
npy_directory = 'D:\\Academics\\Sem6\\Deep Learning\\DL_A2\\npy\\'
my_data = np.empty((1))
flag = False
for ix in tqdm(os.listdir(root)): 
    folder = os.path.join(root, ix)
    print(folder)
    read = lambda imname: np.asarray((Image.open(imname).convert("L")).resize((320, 240), Image.ANTIALIAS))
    ims = [read(os.path.join(folder, filename)) for filename in os.listdir(folder)]
    im_array = np.array(ims, dtype='uint8')    
    
    ch1_arr = np.empty((1))
    ch2_arr = np.empty((1))
    ch3_arr = np.empty((1))
    ch4_arr = np.empty((1))
    ch5_arr = np.empty((1))
    ch6_arr = np.empty((1))
    
    for i in tqdm(range(im_array.shape[0])):
        img1 = im_array[i]
        if(i !=im_array.shape[0] - 1):
            img2 = im_array[i+1]
        else:
            img2 = im_array[i]
        ch1, ch2 = getFlow1(img1, img2)
        ch3 = getFlow2(img1, img2)
        ch4 = getCanny(img1, 50, 50)
        ch5 = getFourier(img1)
        ch6 = getJET(img1)
        
        if(i == 0):
            ch1_arr = np.expand_dims(ch1, axis = 0)
        else:
            ch1_arr = np.concatenate((ch1_arr, np.expand_dims(ch1, axis = 0)), axis = 0)
        
        if(i == 0):
            ch2_arr = np.expand_dims(ch2, axis = 0)
        else:
            ch2_arr = np.concatenate((ch2_arr, np.expand_dims(ch2, axis = 0)), axis = 0)
            
        if(i == 0):
            ch3_arr = np.expand_dims(ch3, axis = 0)
        else:
            ch3_arr = np.concatenate((ch3_arr, np.expand_dims(ch3, axis = 0)), axis = 0)
            
        if(i == 0):
            ch4_arr = np.expand_dims(ch4, axis = 0)
        else:
            ch4_arr = np.concatenate((ch4_arr, np.expand_dims(ch4, axis = 0)), axis = 0)
            
        if(i == 0):
            ch5_arr = np.expand_dims(ch5, axis = 0)
        else:
            ch5_arr = np.concatenate((ch5_arr, np.expand_dims(ch5, axis = 0)), axis = 0)
            
        if(i == 0):
            ch6_arr = np.expand_dims(ch6, axis = 0)
        else:
            ch6_arr = np.concatenate((ch6_arr, np.expand_dims(ch6, axis = 0)), axis = 0)
            
    im_array = np.expand_dims(im_array, axis = 3)
    ch1_arr = np.expand_dims(ch1_arr, axis = 3)
    ch2_arr = np.expand_dims(ch2_arr, axis = 3)
    ch3_arr = np.expand_dims(ch3_arr, axis = 3)
    ch4_arr = np.expand_dims(ch4_arr, axis = 3)
    ch5_arr = np.expand_dims(ch5_arr, axis = 3)
    ch6_arr = np.expand_dims(ch6_arr, axis = 3)
    
    
    
    arr = np.concatenate((im_array, ch1_arr), axis = 3)
    arr = np.concatenate((arr, ch2_arr), axis = 3)
    arr = np.concatenate((arr, ch3_arr), axis = 3)
    arr = np.concatenate((arr, ch4_arr), axis = 3)
    arr = np.concatenate((arr, ch5_arr), axis = 3)
    arr = np.concatenate((arr, ch6_arr), axis = 3)
    
    
    if(flag == False):
        my_data = np.expand_dims(arr, axis = 0)
        flag = True
    else:
        my_data = np.concatenate((my_data, np.expand_dims(arr, axis = 0)), axis = 0)
    
    np.save(npy_directory+ix, arr)
    
