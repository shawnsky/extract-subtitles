# -*- coding: utf-8 -*-

import cv2
import operator
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.signal import argrelextrema
from PIL import Image
import pytesseract
import PIL.ImageOps

#print(sys.executable)
#Setting fixed threshold criteria
USE_THRESH = False
#fixed threshold value
THRESH = 0.6
#Setting fixed threshold criteria
USE_TOP_ORDER = False
#Setting local maxima criteria
USE_LOCAL_MAXIMA = True
#Number of top sorted frames
NUM_TOP_FRAMES = 20

#Adjust crop area mode
ADJUST_MODE = True

#Video path of the source file
videopath = sys.argv[1]
#Directory to store the processed frames
dir = "frames/"
#smoothing window size
len_window = int(sys.argv[2])

#language
LANG='chi_sim'



#set cropper parameters

#left_padding
x=150
#top_padding
y=430
#window_width
w=600
#window_height
h=40


def smooth(x, window_len=13, window='hanning'):
    print(len(x), window_len)
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.") 

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[2 * x[0] - x[window_len:1:-1],
              x, 2 * x[-1] - x[-1:-window_len:-1]]
    #print(len(s))

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]

#Class to hold information about each frame


class Frame:
    def __init__(self, id, frame, value):
        self.id = id
        self.frame = frame
        self.value = value

    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return self.id == other.id and self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)


def rel_change(a, b):
   x = (b - a) / max(a, b)
   print(x)
   return x


def ocr_im(name):
    global ADJUST_MODE
    im=Image.open(dir + name)
    inverted_im=PIL.ImageOps.invert(im)
    #inverted_im.show()
    croped_im=inverted_im.crop((x,y,x+w,y+h))
    if ADJUST_MODE:
        croped_im.show()
        ADJUST_MODE = False
    text=pytesseract.image_to_string(croped_im, LANG)
    return text
    

#Print infos
print("[Video Path] " + videopath)
print("[Frame Directory] " + dir)
print("[Subtitles Language] " + LANG)
print("Extracting key frames, waiting...")


cap = cv2.VideoCapture(str(videopath))


curr_frame = None
prev_frame = None

frame_diffs = []
frames = []
ret, frame = cap.read()
i = 1

while(ret):
    luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
    curr_frame = luv
    if curr_frame is not None and prev_frame is not None:
        #logic here
        diff = cv2.absdiff(curr_frame, prev_frame)
        count = np.sum(diff)
        frame_diffs.append(count)
        frame = Frame(i, frame, count)
        frames.append(frame)
    prev_frame = curr_frame
    i = i + 1
    ret, frame = cap.read()
"""
    cv2.imshow('frame',luv)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
"""
cap.release()
#cv2.destroyAllWindows()
last_subtitle = ""

if USE_TOP_ORDER:
    # sort the list in descending order
    frames.sort(key=operator.attrgetter("value"), reverse=True)
    for keyframe in frames[:NUM_TOP_FRAMES]:
        name = "frame_" + str(keyframe.id) + ".jpg"
        cv2.imwrite(dir + "/" + name, keyframe.frame)

if USE_THRESH:
    for i in range(1, len(frames)):
        if (rel_change(np.float(frames[i - 1].value), np.float(frames[i].value)) >= THRESH):
            #print("prev_frame:"+str(frames[i-1].value)+"  curr_frame:"+str(frames[i].value))
            name = "frame_" + str(frames[i].id) + ".jpg"
            cv2.imwrite(dir + "/" + name, frames[i].frame)


if USE_LOCAL_MAXIMA:
    diff_array = np.array(frame_diffs)
    sm_diff_array = smooth(diff_array, len_window)
    frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]
    for i in frame_indexes:
        name = "frame_" + str(frames[i - 1].id) + ".jpg"
        #print(dir+name)
        cv2.imwrite(dir + name, frames[i - 1].frame)


        text = ocr_im(name)
        #Check for repeated subtitles 
        if text != last_subtitle:
            last_subtitle=text
            print(text)
        #Delete ocred frame images
        os.remove(dir + name)



#Draw plot
# plt.figure(figsize=(40, 20))
# plt.locator_params(numticks=100)
# plt.stem(sm_diff_array)
# plt.savefig(dir + 'plot.png')






