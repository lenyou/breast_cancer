import cv2
import imutils
import matplotlib.pyplot as plt
import random


def rotate_3D():

    
    
    return 



def aug(imgs):
    for im in imgs:
        im_ = im[0,:,:,:]
        a = random.randint(0,360)
        rotated = imutils.rotate(im_,a) 
        plt.imshow(rotated)
        plt.show()
    return 