import cv2
import numpy as np
import time
from PIL import Image
import math




def seed_pixels(stroke_img,img):
    # making two array
    array_1 = np.array(stroke_img)
    array_2 = np.array(img)
    # For blue and red in stroke img
    blue = np.array([6, 0, 255])
    red = np.array([255, 0, 0])
    C = []
    D = []
    m=0
    n=0


    for i in range (0,len(array_1)):
        for j in range (0,len(array_1[0])):
            if (array_1[i][j][0] == blue[0] and array_1[i][j][1] == blue[1] and array_1[i][j][2] == blue[2] ):
                C = np.append(C, [[array_2[i][j][0], array_2[i][j][1], array_2[i][j][2]]] )
                m = m+1
            elif (array_1[i][j][0] == red[0] and array_1[i][j][1] == red[1] and array_1[i][j][2] == red[2]):
                D = np.append(D, [[array_2[i][j][0], array_2[i][j][1], array_2[i][j][2]]] )
                n = n+1

    back_g = C.reshape(m,3)
    for_g = D.reshape(n,3)

    return for_g, back_g




def get_index(arr, cent):
    imd = []
    for w in range(0, len(arr)):
        dis = []
        for i in range(len(cent)):
            d = arr[w] - cent[i]
            dd = (np.linalg.norm(d))
            dis = np.append(dis, dd)
        imd = np.append(imd, np.argmin(dis))
    return imd




def generate_centroids(k,g):
    n = 0
    centroid=[]
    while n < k*3:
        r = np.random.randint(np.amin(np.amin(g)), np.amax(np.amax(g)))
        if r not in centroid:
            n += 1
            centroid=np.append(centroid,r)
    g_centroid = centroid.reshape(k,3)
    return g_centroid



def update_centroid(arr, ind, k):
    avg = []
    for i in range(k):
        hh = []
        m = 0
        for j in range(len(arr)):
            if ind[j] == i:
                hh = np.append(hh, arr[j])
                m += 1
        if len(hh) != 0:
            hh = hh.reshape(m, 3)
            avg = np.append(avg, (sum(hh) / len(hh)))

        else:
            avg = np.append(avg, arr[i])

    avg = avg.reshape(k, 3)
    return avg



def kmean(k,g):
    centroids = generate_centroids(k, g)
    cent = np.empty_like(centroids)
    iterations = 0
    while not np.array_equal(cent, centroids):
        if iterations <= 200:
            iterations += 1
            cent = centroids
            index = get_index(g, centroids)
            centroids = update_centroid(g, index, k)
        else:
            cent = centroids
    print('total number of iterations:')
    print(iterations)
    return centroids, index



def weight(cent,index):
    tsp=len(index)
    k=len(cent)
    num=[]
    for i in range (k):
        l=0
        for j in range (tsp):
            if index[j]==i:
                l+=1
        num=np.append(num,l)
    weigh=num/tsp
    return weigh



def likelihood(img, centroid_f, index_f, centroid_b, index_b):
    imag1 = img
    im1 = np.array(imag1)
    print(im1.shape)

    fww = weight( centroid_f, index_f)
    bww = weight(centroid_b, index_b)

    b = np.zeros([len(im1), len(im1[0]), 3])
    p1 = np.zeros([1, 3])


    for i in range(0, len(im1)):
        for j in range(0, len(im1[0])):
            p1[0][0] = im1[i][j][0]
            p1[0][1] = im1[i][j][1]
            p1[0][2] = im1[i][j][2]

            add = 0
            for k in range(0, len(centroid_f)):
                wk = fww[k]
                norm1 = (np.linalg.norm(p1 - centroid_f[k]))
                expo = math.exp(-(norm1))
                add = add + (wk * expo)
            fore_ground = add

            add_1 = 0
            for m in range(0, len(centroid_b)):
                wk_1 = bww[m]
                norm_1 = (np.linalg.norm(p1 - centroid_b[m]))
                expo_1 = math.exp(-norm_1)
                add_1 = add_1 + (wk_1 * expo_1)
            back_ground = add_1

            if fore_ground > back_ground:
                b[i][j] = [255, 255, 255]

            else:
                b[i][j] = [0, 0, 0]
    return b



def display(b,img):

    img_1 = np.array(img.copy())
    img_2 = np.array(img.copy())

    for i in range(len(b)):
        for j in range(len(b[0])):
            if b[i][j][0] == 0:
                img_1[i][j] = [0,0,0]
    for i in range(len(b)):
        for j in range(len(b[0])):
            if b[i][j][0] != 0:
                img_2[i][j] = [0,0,0]

    cv2.imshow('image', img)
    cv2.imshow('image 1', img_1)
    cv2.imshow('image 2', img_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def Segmentation(img,stroke_img ,k):
    start = time.time()
    s_img = stroke_img
    for_g, back_g = seed_pixels(s_img, img)

    centroid_f, index_f = kmean(k, for_g)
    centroid_b, index_b = kmean(k, back_g)

    b = likelihood(img, centroid_f, index_f, centroid_b, index_b)
    display(b, img)

    end = time.time()
    print("process time in minutes")
    print((end-start)/60)




def main():

    stroke_img1 = Image.open(r'C:\Users\k-ali\Desktop\FYP\fyp\ML sem\dip\assign 2\dance stroke 1.png')
    stroke_img2 = Image.open(r'C:\Users\k-ali\Desktop\FYP\fyp\ML sem\dip\assign 2\dance stroke 2.png')
    original_img = cv2.imread(r'C:\Users\k-ali\Desktop\FYP\fyp\ML sem\dip\assign 2\dance.PNG')
    N = 5

    Segmentation(original_img, stroke_img1, N)
    Segmentation(original_img, stroke_img2, N)





if __name__ == '__main__':
    main()

