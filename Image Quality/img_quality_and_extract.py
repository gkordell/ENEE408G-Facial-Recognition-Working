# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 18:45:36 2018

@author: pydea
"""
import scipy.io as sio
import imageio as imgio
import numpy as np
import dlib
import sys
import matplotlib.pyplot as plt

# Useful example code: https://github.com/davisking/dlib/blob/master/python_examples/face_detector.py

# Requirements for the image to be high enough quality (I am making these up)
# - dlib must be able to detect a face
# - the frame of the face found by dlib must be >30% of the area of the image
# - the dlib-extracted "score" for the face must be greater than 2

# argv 1 should be the saved JPEG Image of the input face
if __name__ == "__main__":
    #filename = sys.argv[1]
    filename = 'Peter_Deaville_0001.JPEG'
    #filename = 'Adam_Sandler_0001.jpg'
    #filename = 'Al_Pacino_0001.jpg'
    #filename = 'Aaron_Sorkin_0001.jpg'
    face_good = False   # defaults to not a good face
    detector = dlib.get_frontal_face_detector()
    
    img = imgio.imread(filename)
    total_im_area = img.shape[0]*img.shape[1]
        
        # detector.run takes three args, first is img, second is upsample factor, third is threshold (lower means more detections, 0 is good)
    dets, scores, idx = detector.run(img, 1, -1)     # I'm not sure what idx does - it doesn't seem necessary for us
    if len(dets) > 0:    # if it has found a face at all
        d = dets[0]
        face_area = (d.right()-d.left())*(d.bottom()-d.top())
        shape = np.asarray([d.top(), d.bottom(), d.left(), d.right()])
        pct_coverage = face_area/total_im_area
        if pct_coverage > 0.1:
            if scores[0] > 1:
                face_good = True       
        if face_good == True:
            print('2')      #face is present AND good
        else:
            print('1')      #face is present AND not good
    else:
        print('0')          #face is NOT detected
        