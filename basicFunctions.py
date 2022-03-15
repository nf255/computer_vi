import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#########################################
#
# Reading Images
#
# img = cv.imread('Katze.jpg')
# cv.imshow('Cat', img)
# cv.waitKey(0)
#
#########################################
#
# Reading Videos
#
# capture = cv.VideoCapture('homm.mp4')
#
# while True:
#     isTrue, frame = capture.read()
#     cv.imshow('Video', frame)
#
#     if cv.waitKey(20) & 0xFF==ord('d'):
#         break
#
# capture.release()
# cv.destroyAllWindows()
#
#########################################
#
# Rescale Images, Videos, Live Videos
#
# def rescaleFrame(frame, scale=0.75):
#     width = int(frame.shape[1] * scale)
#     height = int(frame.shape[0] * scale)
#     dimensions = (width, height)
#     return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
#
#########################################
#
# Rescale Live Videos
#
# def changeRes(width, height):
#     capture.set(3, width)
#     capture.set(4, height)
#
#########################################
#
# Paint the entire image in a certain color
#
# blank = np.zeros((500, 500, 3), dtype='uint8')
# blank[200:300, 300:400] = (0, 255, 255)
# cv.imshow('Green', blank)
# cv.waitKey(0)
#
#########################################
#
# Draw a rectangle, a circle, a line and text
#
# blank = np.zeros((500, 500, 3), dtype='uint8')
#
# cv.rectangle(blank, (0, 0), (250, 250), (0, 255, 0), thickness=2)
# cv.circle(blank, (250, 250), 50, (0, 0, 255), thickness=-1)
# cv.line(blank, (0, 0), (250, 250), (0, 0, 255), thickness=2)
# cv.putText(blank, 'Hello World', (150, 150), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 2)
#
# cv.imshow('rect', blank)
# cv.waitKey(0)
#
#########################################
#
# Image grayscale, blurring, edge cascade, dilate, erode, resize, crop
#
# img = cv.imread('Katze.jpg')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# blur = cv.GaussianBlur(gray, (3, 3), cv.BORDER_DEFAULT)
# canny = cv.Canny(blur, 125, 175)
# dilated = cv.dilate(canny, (7, 7), iterations=3)
# eroded = cv.erode(dilated, (3, 3), iterations=1)
# resized = cv.resize(eroded, (500, 500), interpolation=cv.INTER_CUBIC)
# cropped = resized[50:200, 200:400]
# cv.imshow('cropped', cropped)
# cv.waitKey(0)
#
#########################################
#
# Image translation
#
# img = cv.imread('Katze.jpg')
# def translate(img, x, y):
#     transMat = np.float32([[1, 0, x], [0, 1, y]])
#     dimensions = (img.shape[1], img.shape[0])
#     return cv.warpAffine(img, transMat, dimensions)
# translated = translate(img, 100, 100)
# cv.imshow('result', translated)
# cv.waitKey(0)
#
#########################################
#
# Image rotation
#
# img = cv.imread('Katze.jpg')
# def rotate(img, angle, rotPoint=None):
#     (height, width) = img.shape[:2]
#     if rotPoint is None:
#         rotPoint = (width//2, height//2)
#     rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
#     dimensions = (width, height)
#     return cv.warpAffine(img, rotMat, dimensions)
# rotated = rotate(img, 45)
# cv.imshow('result', rotated)
# cv.waitKey(0)
#
#########################################
#
# Image resizing, flipping
#
# img = cv.imread('Katze.jpg')
# resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
# flipped = cv.flip(img, 0)
# cv.imshow('result', flipped)
# cv.waitKey(0)
#
#########################################
#
# contour detection
#
# img = cv.imread('Katze.jpg')
#
# blank = np.zeros(img.shape, dtype='uint8')
#
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# # blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
# canny = cv.Canny(gray, 125, 175)
# cv.imshow('canny', canny)
#
# ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
#
# contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# print(f'{len(contours)} contour(s) found!')
#
# cv.drawContours(blank, contours, -1, (0, 0, 255), 1)
# cv.imshow('Contours drawn', blank)
# cv.waitKey(0)
#
#########################################
#
# Color Spaces
#
# img = cv.imread('Katze.jpg')
#
# # BGR to Grayscale
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('gray', gray)
#
# # BGR to HSV
# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# cv.imshow('HSV', hsv)
#
# # BGR to L*a*b
# lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
# cv.imshow('LAB', lab)
#
# # BGR to RGB
# rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# cv.imshow('RGB', rgb)
#
# cv.waitKey(0)
#
#########################################
#
# Split and merge color channels
#
# img = cv.imread('Katze.jpg')
#
# blank = np.zeros(img.shape[:2], dtype='uint8')
# b, g, r = cv.split(img)
# blue = cv.merge([b, blank, blank])
# green = cv.merge([blank, g, blank])
# red = cv.merge([blank, blank, r])
#
# cv.imshow('blue', blue)
# cv.imshow('green', green)
# cv.imshow('red', red)
#
# merged = cv.merge([b, g, r])
# cv.imshow('merged', merged)
# cv.waitKey(0)
#
#########################################
#
# Blurring
#
# img = cv.imread('Katze.jpg')
#
# # Average blur
# average = cv.blur(img, (3, 3))
#
# # Gaussian blur
# gauss = cv.GaussianBlur(img, (3, 3), 0)
#
# # Median blur
# median = cv.medianBlur(img, 3)
#
# # Bilateral blur
# bilateral = cv.bilateralFilter(img, 10, 35, 25)
#
# cv.imshow('result', bilateral)
# cv.waitKey(0)
#
#########################################
#
# Bitwise operations
#
# blank = np.zeros((400, 400), dtype='uint8')
#
# rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
# circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)
#
# bitwise_and = cv.bitwise_and(rectangle, circle)
# cv.imshow('AND', bitwise_and)
#
# bitwise_or = cv.bitwise_or(rectangle, circle)
# cv.imshow('OR', bitwise_or)
#
# bitwise_xor = cv.bitwise_xor(rectangle, circle)
# cv.imshow('XOR', bitwise_xor)
#
# bitwise_not = cv.bitwise_not(rectangle)
# cv.imshow('NOT', bitwise_not)
# cv.waitKey(0)
#
#########################################
#
# Masking
#
# img = cv.imread('Katze.jpg')
#
# blank = np.zeros(img.shape[:2], dtype='uint8')
# mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
# masked = cv.bitwise_and(img, img, mask=mask)
# cv.imshow('masked', masked)
#
# cv.waitKey(0)
#
#########################################
#
# Grayscale histogram
#
# img = cv.imread('Katze.jpg')
#
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# blank = np.zeros(gray.shape[:2], dtype='uint8')
#
# circle = cv.circle(blank, (gray.shape[1]//2, gray.shape[0]//2), 100, 255, -1)
# mask = cv.bitwise_and(gray, gray, mask=circle)
# gray_hist = cv.calcHist([gray], [0], mask, [256], [0, 256])
#
# plt.figure()
# plt.title('Grayscale Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of pixels')
# plt.plot(gray_hist)
# plt.xlim([0, 256])
# plt.ylim([0, 5000])
# plt.show()
#
# cv.imshow('result', mask)
# cv.waitKey(0)
#
#########################################
#
# Color Histogram
#
# img = cv.imread('Katze.jpg')
# blank = np.zeros(img.shape[:2], dtype='uint8')
#
# mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
# masked = cv.bitwise_and(img, img, mask=mask)
#
# plt.figure()
# plt.title('Color Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of pixels')
# colors = ('b', 'g', 'r')
# for i, col in enumerate(colors):
#     hist = cv.calcHist([img], [i], mask, [256], [0, 256])
#     plt.plot(hist, color=col)
#     plt.xlim([0, 256])
# plt.show()
#
# cv.imshow('result', masked)
# cv.waitKey(0)
#
#########################################
#
# Thresholding
#
# img = cv.imread('Katze.jpg')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
# # Simple thresholding
#
# threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
# cv.imshow('Simple thresholded', thresh)
#
# threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
# cv.imshow('Simple thresholded', thresh_inv)
#
# # Adaptive Thresholding
#
# adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 5)
# cv.imshow('Adaptive thresholded', adaptive_thresh)
#
# cv.waitKey(0)
#
#########################################
#
# Edge detection
#
# img = cv.imread('Katze.jpg')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
# # Laplacian
#
# lap = cv.Laplacian(gray, cv.CV_64F)
# lap = np.uint8(np.absolute(lap))
# cv.imshow('Laplacian', lap)
#
# # Sobel
#
# sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
# sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
# combined_sobel = cv.bitwise_or(sobelx, sobely)
#
# # Canny
#
# canny = cv.Canny(gray, 150, 175)
#
# cv.imshow('Sobel X', sobelx)
# cv.imshow('Sobel Y', sobely)
# cv.imshow('Combined Sobel', combined_sobel)
# cv.imshow('Canny', canny)
# cv.waitKey(0)
