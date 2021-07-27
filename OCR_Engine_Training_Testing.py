import cv2
import numpy as np

# ######   training part    ###############
samples = np.loadtxt('/home/murthy/PycharmProjects/python/OCR_Engine/alphabet_data/generalsamples.data', np.float32)
responses = np.loadtxt('/home/murthy/PycharmProjects/python/OCR_Engine/alphabet_data/generalresponses.data', np.float32)
responses = responses.reshape((responses.size, 1))

model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

# ############################ testing part  #########################

im = cv2.imread('/home/murthy/PycharmProjects/python/OCR_Engine/alphabet_data/alpha2.png')
out = np.zeros(im.shape, np.uint8)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt) > 50:
        [x, y, w, h] = cv2.boundingRect(cnt)
        print("Height:::", h)
        if h > 25:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = thresh[y: y + h, x: x + w]
            roismall = cv2.resize(roi, (10, 10))
            roismall = roismall.reshape((1, 100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)
            # string = str(int((results[0][0])))
            string = chr(int((results[0][0])))
            print("string::::::", string)
            cv2.putText(out, string, (x, y + h), 0, 1, (255, 255, 255))

cv2.imwrite("/home/murthy/PycharmProjects/python/OCR_Engine/alphabet_data/predict_result.png", out)
cv2.imwrite("/home/murthy/PycharmProjects/python/OCR_Engine/alphabet_data/input.png", im)
cv2.imshow('im', im)
cv2.imshow('out', out)
cv2.waitKey(0)


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Load the data, converters convert the letter to a number
# data = np.loadtxt('/home/narasimha/PycharmProjects/python/OCR_Engine/alphabet_data/letter-recognition.data',
#                   dtype='float32', delimiter=',', converters={0: lambda ch: ord(ch)-ord('A')})
#
# # split the data to two, 10000 each for train and test
# train, test = np.vsplit(data, 2)
#
# # split trainData and testData to features and responses
# responses, trainData = np.hsplit(train, [1])
# labels, testData = np.hsplit(test, [1])
#
# # Initiate the kNN, classify, measure accuracy.
# knn = cv2.ml.KNearest_create()
# knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
# # ret, result, neighbours, dist = knn.findNearest(testData, k=5)
#
# # ######### predicting the characters #################
# im = cv2.imread('/home/narasimha/PycharmProjects/python/OCR_Engine/alphabet_data/foo-page0.jpg')
# out = np.zeros(im.shape, np.uint8)
# gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
#
# images, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#
# for cnt in contours:
#     if cv2.contourArea(cnt) > 50:
#         print("entering if contour")
#         [x, y, w, h] = cv2.boundingRect(cnt)
#         print("height", h)
#         if h > 10:
#             print("entering if height")
#             cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             roi = thresh[y: y + h, x: x + w]
#             roismall = cv2.resize(roi, (10, 10))
#             roismall = roismall.reshape((1, 100))
#             roismall = np.float32(roismall)
#             retval, results, neigh_resp, dists = knn.findNearest(roismall, k=5)
#             string = str(int((results[0][0])))
#             cv2.putText(out, string, (x, y + h), 0, 1, (0, 255, 0))
#
#             correct = np.count_nonzero(results == labels)
#             accuracy = correct*100.0/10000
#             print(accuracy)
#
# cv2.imshow('im', im)
# cv2.imshow('out', out)
# cv2.waitKey(0)
