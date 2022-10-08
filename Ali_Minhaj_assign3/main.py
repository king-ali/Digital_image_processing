

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import glob
import utlis



# For our ease we used glob to see results of every picture in the image directory
imag = glob.glob("Images/*.jpg")
print(np.asarray(imag).shape)

for images in imag:

    images = cv2.imread(images)
    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    image = cv2.Canny(images, 200, 300,
                      apertureSize=3)
    plt.imshow(image, cmap='gray')

    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180)
    hspace, theta, dist = utlis.hough_line(image)
    print("accumulator is ")
    print(hspace)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(hspace)


    h, q, d = utlis.peak_votes(hspace, theta, dist)
    print("accumulator is ")
    print(h)
    angle_list = []

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(images, cmap='gray')
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(np.log(1 + hspace),
                 extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), dist[-1], dist[0]],
                 cmap='gray', aspect=1 / 1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')
    ax[2].imshow(images)
    # ax[2].imshow(image, cmap='gray')

    origin = np.array((0, images.shape[1]))

    for _, angle, dist in zip(*utlis.peak_votes(hspace, theta, dist)):
        angle_list.append(angle)
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        ax[2].plot(origin, (y0, y1), '-r')
    ax[2].set_xlim(origin)
    ax[2].set_ylim((images.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    plt.tight_layout()
    plt.show()

    angles = [a * 180 / np.pi for a in angle_list]
    angle_difference = np.max(angles) - np.min(angles)
    print(180 - angle_difference)





