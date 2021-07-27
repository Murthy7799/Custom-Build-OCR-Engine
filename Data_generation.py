import sys
import numpy as np
import cv2

# im = cv2.imread('/home/narasimha/PycharmProjects/python/OCR_Engine/data2/sometext3.png')
im = cv2.imread('/home/murthy/PycharmProjects/python/OCR_Engine/alphabet_data/randomtext.png')
# im = cv2.resize(im, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)
im3 = im.copy()

# Converting the image to grey using cv2 methods and thresholding the image

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

# ################      Now finding Contours      ###################

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Generating numpy arrays to feed the data generated from the images for training and testing.
samples = np.empty((0, 100), np.float32)
responses = []
# Generating a list of ASCII values of 48 to 123 to identify the values in the image typing from keyboard and fill it.
keys = [i for i in range(48, 123)]

# Looping through contours for getting the area and the x,y,h,w values of the individual text
for cnt in contours:
    # print("cnt:::::", cnt)
    if cv2.contourArea(cnt) > 50:
        [x, y, w, h] = cv2.boundingRect(cnt)
        print("height::", h)

        if h > 25:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
            roi = thresh[y:y + h, x:x + w]
            roismall = cv2.resize(roi, (10, 10))
            cv2.imshow('norm', im)
            key = cv2.waitKey(0)
            print("key:::", key)

            print("keys::::", keys)
            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(key)
                sample = roismall.reshape((1, 100))
                samples = np.append(samples, sample, 0)

# print("samples:::", samples)
# print("responses1:::", responses)
responses = np.array(responses)
responses = responses.reshape((responses.size, 1))
print("Data Generation complete")
# print("responses:::", responses)

samples = np.float32(samples)
responses = np.float32(responses)
print("responses2:::", responses)
print("samples2:::", samples)

cv2.imwrite("/home/murthy/PycharmProjects/python/OCR_Engine/alphabet_data/train_result.png", im)
np.savetxt('/home/murthy/PycharmProjects/python/OCR_Engine/alphabet_data/generalsamples.data', samples)
np.savetxt('/home/murthy/PycharmProjects/python/OCR_Engine/alphabet_data/generalresponses.data', responses)
