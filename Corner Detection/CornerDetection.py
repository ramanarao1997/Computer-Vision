import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy import signal
from PIL import Image
from scipy import ndimage as ndi
import cv2

#---------------------------------- Moravec Corner Detection -------------------------------------------

# function 1
def exact_Ep_d(img, pixel, angle):
  x, y = pixel[0 : 2]

  directions = { 0: (1, 0), 45: (1, 1), 90: (0, 1), 135: (-1, 1), 180: (-1, 0) }
  u, v = directions[angle]

  Epd = img.getpixel( (x + u, y + v)) - img.getpixel( (x, y) )

  return Epd ** 2

### function 2
def exact_Ep(img, pixel, angles = (0, 45, 90, 135, 180) ):
  x, y = pixel[0 : 2]
  minEp = 9999999

  for angle in angles:
    temp = exact_Ep_d(img, pixel, angle)
    if(temp < minEp):
        minEp = temp

  return minEp

### function 3
def moravec_corners(img, threshold, kernel = [[0,0,0], [0,1,0], [0,0,0]]):
    corners = []

    np_energy_img = np.copy(img)
    np_corners_img = np.copy(img)

    energy_img = Image.fromarray(np_energy_img)
    corners_img = Image.fromarray(np_corners_img)

    w, h = img.size

    # run above function for each pixel of the image
    for x in range(1, w - 1):
        for y in range(1, h - 1):
            energy_img.putpixel((x, y), exact_Ep(img, (x, y)))

    energy_img = signal.convolve2d(energy_img, kernel, mode = 'same', boundary = 'fill')

    # careful things get flipped
    for x in range(1, energy_img.shape[1] - 1):
        for y in range(1, energy_img.shape[0] - 1):
            if(energy_img[y, x] > threshold):
                corners.append((y, x))

    return corners

#------------------------------- Harris Corner Detection ----------------------------------------------

### function 4
def harris_energy(img, kernel):
    sobelFilterX = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    sobelFilterY = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])

    I_x = signal.convolve2d(img, sobelFilterX , mode='same')
    I_y = signal.convolve2d(img, sobelFilterY , mode='same')

    Ixx = ndi.gaussian_filter(I_x**2, sigma = 1)
    Ixy = ndi.gaussian_filter(I_y*I_x, sigma = 1)
    Iyy = ndi.gaussian_filter(I_y**2, sigma = 1)

    return [[Ixx, Ixy], [Ixy, Iyy]]

# function 5
def harris_corners(img, kernel = "Gaussian", lamda = 1 , threshold = 0):
    M = harris_energy(img, kernel)

    Ixx = M[0][0]
    Ixy = M[0][1]
    Iyy = M[1][1]

    detM = Ixx * Iyy - Ixy ** 2
    traceM = Ixx + Iyy

    harris_response = detM - lamda * traceM
    maxR = np.amax(harris_response)
    normalized_harris_response = np.copy(harris_response).astype(float)

    for i in range(len(harris_response)):
        for j in range(len(harris_response[0])):
            normalized_harris_response[i, j] = harris_response[i, j] / maxR
            
    harris_corners = []

    for rowindex, response in enumerate(normalized_harris_response):
       for colindex, R in enumerate(response):
           if R > threshold:
               harris_corners.append( (rowindex, colindex) )

    return (harris_corners, normalized_harris_response)

#_______________________________________________________________________________________________________________________

# Extra Credit (using separable filter (gaussian))
def harris_energy_separable(img, kernel):
    sobelFilterX = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    sobelFilterY = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])

    I_x = signal.convolve2d(img, sobelFilterX , mode='same')
    I_y = signal.convolve2d(img, sobelFilterY , mode='same')

    Ixx = ndi.gaussian_filter(I_x**2, sigma = 1)
    Ixy = ndi.gaussian_filter(I_y*I_x, sigma = 1)
    Iyy = ndi.gaussian_filter(I_y**2, sigma = 1)

    return [[Ixx, Ixy], [Ixy, Iyy]]

#_______________________________________________________________________________________________________________________