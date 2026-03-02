#!/usr/bin/env python3

import cv2
import time
import imutils
import numpy as np

vs = cv2.VideoCapture(2)
import pytesseract
custom_config = r'--oem 3 --psm 6 outputbase digits'
custom_config = r'--oem 3 --psm 13 outputbase digits'
custom_config = r'--oem 3 --psm 8 outputbase digits'
import matplotlib.pylab as plt
plt.ion()

# allow the camera or video file to warm up
time.sleep(2.0)

data=[]
times=[]
i=0
print('')
print('Started Acquisition - reporting every N seconds')

while True:
    ret, frame = vs.read()
    if frame is None:
        break
    #cv2.imshow("Frame", frame)
    i=i+1
    temp = cv2.cvtColor(frame[115:155,45:115,:], cv2.COLOR_BGR2GRAY)
    temp = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    temp = 255 - temp
    #temp = cv2.GaussianBlur(temp, (3, 3), 1)
    readout=pytesseract.image_to_string(temp, config=custom_config).strip()
    readout = readout.replace('(','').replace('<','').replace('>','')
    if readout.find('.')>0:
        readout = readout[:readout.find('.')+2]
        try:
            readout = float(readout)
            if i%100==0:
                print(readout)
            data.append(readout)
            times.append(time.time())
            with open('/dev/shm/flir.txt','a') as fp:
                fp.write('%.1f,%.1f\n'%(time.time(), readout))
        except:
            print('nan: ', readout)

vs.release()

