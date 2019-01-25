# -*- coding: utf-8 -*-

import cv2
import operator
import numpy as np
import matplotlib.pyplot as plt
import sys
from PIL import Image
import pytesseract
import PIL.ImageOps

#set cropper parameters
# 我不是药神 100 410 600 40
# minecraft 80,285,550,38
# 狗十三 150 430 600 40
#left_padding
x=150
#top_padding
y=430
#window_width
w=600
#window_height
h=40


imagePath = sys.argv[1]

# src=cv2.imread(imagePath)
# cv2.imshow("Before", src)
# cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)

im=Image.open(imagePath)
# image=np.asarray(im)
# height,width=image.size
# print(height)
# image2=image.copy()
# for i in range(height):
#     for j in range(width):
#         image2[i,j]=(255-image[i,j])

# image2.show()

#contrast
#contrasted_im=PIL.ImageOps.autocontrast(im, cutoff=0)
#contrasted_im.show()

#gray
#grayed_im=PIL.ImageOps.grayscale(im)
#grayed_im.show()

#invert
inverted_im=PIL.ImageOps.invert(im)
#inverted_im.show()


croped_im=inverted_im.crop((x,y,x+w,y+h))
croped_im.show()
text=pytesseract.image_to_string(croped_im, lang='chi_sim')
print(text)