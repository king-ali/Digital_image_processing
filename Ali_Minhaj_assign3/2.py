import cv2 as cv
import numpy as np
import glob

def Hough_line(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv.imshow('houghlines3.jpg', img)
        cv.waitKey(0)

    else:
        print("No line detected in image")

# For our ease we used glob to see results of every picture in the image directory
images = glob.glob("Images/*.jpg")
for img in images:
    img = cv.imread(img)
    Hough_line(img)
