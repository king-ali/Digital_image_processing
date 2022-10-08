import pylab as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def imhist(im):
	m, n = im.shape
	h = [0.0] * 256
	for i in range(m):
		for j in range(n):
			h[im[i, j]]+=1
	return np.array(h)/(m*n)

def cumsum(h):

	return [sum(h[:i+1]) for i in range(len(h))]

def histeq(im):

	h = imhist(im)
	cdf = np.array(cumsum(h))
	sk = np.uint8(255 * cdf)
	s1, s2 = im.shape
	Y = np.zeros_like(im)

	for i in range(0, s1):
		for j in range(0, s2):
			Y[i, j] = sk[im[i, j]]
	H = imhist(Y)

	return Y , h, H, sk





img = cv2.imread('data/hw1_dark_road_2.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

new_img, h, new_h, sk = histeq(img)


plt.subplot(121)
plt.imshow(img)
plt.title('original image')
plt.set_cmap('gray')

plt.subplot(122)
plt.imshow(new_img)
plt.title('hist. equalized image')
plt.set_cmap('gray')


fig = plt.figure()
fig.add_subplot(221)
plt.plot(h)
plt.title('Original histogram')

fig.add_subplot(222)
plt.plot(new_h)
plt.title('New histogram')

fig.add_subplot(223)
plt.plot(sk)
plt.title('Transfer function')



clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized = clahe.apply(img)
h = imhist(equalized)
fig = plt.figure()
fig.add_subplot()
plt.plot(h)
plt.title('Histogram of modified image')
# Experimenting with different values to see result
cv2.imshow("Input", img)
cv2.imshow("CLAHE", equalized)
clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(8, 8))
equalized = clahe.apply(img)
equalized = clahe.apply(img)
h = imhist(equalized)
fig = plt.figure()
fig.add_subplot()
plt.plot(h)
plt.title('Histogram of modified image 2')

cv2.imshow("CLAHE 2", equalized)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()