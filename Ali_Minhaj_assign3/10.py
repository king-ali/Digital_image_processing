import numpy as np
import matplotlib.pyplot as plt
import cv2
import utils


def houghLine(image):

    Ny = image.shape[0]
    Nx = image.shape[1]
    Maxdist = int(np.round(np.sqrt(Nx ** 2 + Ny ** 2)))
    thetas = np.deg2rad(np.arange(-90, 90))
    rs = np.linspace(-Maxdist, Maxdist, 2 * Maxdist)

    accumulator = np.zeros((2 * Maxdist, len(thetas)))

    for y in range(Ny):
        for x in range(Nx):
            if image[y, x] > 0:
                for k in range(len(thetas)):
                    r = x * np.cos(thetas[k]) + y * np.sin(thetas[k])
                    accumulator[int(r) + Maxdist, k] += 1

    return accumulator, thetas, rs



images = cv2.imread('Images/airport_inside_0098.jpg')
images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
image = cv2.Canny(images, 50, 150, apertureSize=3)
accumulator, thetas, rhos = houghLine(image)


idx, theta, rho = utils.peak_votes(accumulator, thetas, rhos)
for rho, theta in zip(rhos, thetas):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(images, (x1, y1), (x2, y2), (0, 0, 255), 7)
# cv2.imshow('images', images)
# cv2.waitKey(0)


#
# plt.figure('Original Image')
# plt.imshow(image)
# plt.set_cmap('gray')
# plt.figure('Hough Space')
# plt.imshow(accumulator)
# plt.set_cmap('gray')
# plt.show()




