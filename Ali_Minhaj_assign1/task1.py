import cv2
import numpy as np
import matplotlib.pyplot as plt


# Part A

img1 = cv2.imread('data/hw1_atrium.hdr')
img1 = cv2.resize(img1, (0,0), fx=0.6, fy=0.6)
cv2.imshow('image 1', img1)
imgGray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
cv2.imshow('image Gray 1', imgGray1)
img2 = cv2.imread('data/hw1_memorial.hdr')
img2 = cv2.resize(img2, (0,0), fx=0.6, fy=0.6)
imgGray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
cv2.imshow('image Gray 2', imgGray2)
cv2.imshow('image 2', img2)




# Part B
# Trying 4 gamma values for image 1
for gamma in [0.1, 0.5, 1.2, 2.2]:

    gamma_corrected = np.array(255 * (img1 / 255) ** gamma, dtype='uint8')
    cv2.imwrite('gamma_transformed' + str(gamma) + '.jpg', gamma_corrected)

# Trying 4 gamma values for image 2
for gamma in [0.1, 0.5, 1.2, 2.2]:

    gamma_corrected = np.array(255 * (img2 / 255) ** gamma, dtype='uint8')
    cv2.imwrite('gamma_transformed_img2' + str(gamma) + '.jpg', gamma_corrected)

# ao 0.5 value of gamma provides us best details as in 1.2 ans 2.2 value image become darker and for 0.1 image is washed out


# Part c

(B, G, R) = cv2.split(img1)
# show each channel individually
# cv2.imshow("Red", R)
# cv2.imshow("Green", G)
# cv2.imshow("Blue", B)

gamma_corrected1 = np.array(255 * (B / 255) ** 0.5, dtype='uint8')
gamma_corrected2 = np.array(255 * (G / 255) ** 0.5, dtype='uint8')
gamma_corrected3 = np.array(255 * (R / 255) ** 0.5, dtype='uint8')
cv2.imwrite('gamma_transformed1_same' + str(0.5) + '.jpg', gamma_corrected1)
cv2.imwrite('gamma_transformed2_same' + str(0.5) + '.jpg', gamma_corrected2)
cv2.imwrite('gamma_transformed3_same' + str(0.5) + '.jpg', gamma_corrected3)

merged = cv2.merge([gamma_corrected1, gamma_corrected2, gamma_corrected3])

cv2.imshow('merged image with same gamma value', merged)

# Now using different gamma values for different channels


gamma_corrected1 = np.array(255 * (B / 255) ** 2, dtype='uint8')
gamma_corrected2 = np.array(255 * (G / 255) ** 1.5, dtype='uint8')
gamma_corrected3 = np.array(255 * (R / 255) ** 0.5, dtype='uint8')
merged = cv2.merge([gamma_corrected1, gamma_corrected2, gamma_corrected3])
cv2.imshow('merged image with different gamma value', merged)

gamma_corrected1 = np.array(255 * (B / 255) ** 0.5, dtype='uint8')
gamma_corrected2 = np.array(255 * (G / 255) ** 1.5, dtype='uint8')
gamma_corrected3 = np.array(255 * (R / 255) ** 2, dtype='uint8')
merged = cv2.merge([gamma_corrected1, gamma_corrected2, gamma_corrected3])
cv2.imshow('(2) merged image with different gamma value ', merged)

# Similarly for second image

(B, G, R) = cv2.split(img2)

# same gamma value to each channel
gamma_corrected1 = np.array(255 * (B / 255) ** 0.5, dtype='uint8')
gamma_corrected2 = np.array(255 * (G / 255) ** 0.5, dtype='uint8')
gamma_corrected3 = np.array(255 * (R / 255) ** 0.5, dtype='uint8')
merged = cv2.merge([gamma_corrected1, gamma_corrected2, gamma_corrected3])
cv2.imshow('merged image 2 with same gamma value', merged)

# For different values
gamma_corrected1 = np.array(255 * (B / 255) ** 2, dtype='uint8')
gamma_corrected2 = np.array(255 * (G / 255) ** 1.5, dtype='uint8')
gamma_corrected3 = np.array(255 * (R / 255) ** 0.5, dtype='uint8')
merged = cv2.merge([gamma_corrected1, gamma_corrected2, gamma_corrected3])
cv2.imshow('merged image 2 with different gamma value', merged)

gamma_corrected1 = np.array(255 * (B / 255) ** 0.5, dtype='uint8')
gamma_corrected2 = np.array(255 * (G / 255) ** 1.5, dtype='uint8')
gamma_corrected3 = np.array(255 * (R / 255) ** 2, dtype='uint8')
merged = cv2.merge([gamma_corrected1, gamma_corrected2, gamma_corrected3])
cv2.imshow('(2) merged image 2 with different gamma value', merged)

cv2.waitKey(0)
cv2.destroyAllWindows()
