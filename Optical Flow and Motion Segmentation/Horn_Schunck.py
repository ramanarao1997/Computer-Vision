import cv2
import numpy as np
import sys
import warnings
from scipy.ndimage.filters import convolve

desImage = None

def HornSchunck(im1, im2, alpha = 0.0001, N=50):
    lk_params = dict(winSize=(5, 5), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    HSFilter = np.array([[1 / 12, 1 / 6, 1 / 12],
                         [1 / 6, 0, 1 / 6],
                         [1 / 12, 1 / 6, 1 / 12]], float)

    # initialize using L-K
    p0 = cv2.goodFeaturesToTrack(im1, mask=None, **feature_params)
    p1, st, err = cv2.calcOpticalFlowPyrLK(im1, im2, p0, None, **lk_params)
    UV = p1 - p0
    p0 = p0.astype(int)

    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    # initialize the velocities
    U = np.zeros([im1.shape[0], im1.shape[1]])
    V = np.zeros([im1.shape[0], im1.shape[1]])

    for value, vt in zip(p0, UV):
        x = value[0][0]
        y = value[0][1]
        U[y][x] = vt[0][0]
        V[y][x] = vt[0][1]

    [dx, dy, dt] = computeGradients(im1, im2)

    # iterate to refine the velocities U,V
    for i in range(N):
        Un = convolve(U, HSFilter)
        Vn = convolve(V, HSFilter)

        derivatives = (dx * Un + dy * Vn + dt) / (alpha ** 2 + dx ** 2 + dy ** 2)

        U = Un - dx * derivatives
        V = Vn - dy * derivatives

    return U, V

# smoothen the image
def smoothen(imgs, n):
    kernal = np.ones((n, n), np.float32) / (n * n)
    dst = cv2.filter2D(imgs, -1, kernal)
    return dst

# calculating derivatives wrt to X, Y and time
def computeGradients(im1, im2):
    kernelX = np.array([[-1, 1],
                   [-1, 1]]) * .25
    kernelY = np.array([[-1, -1],
                   [1, 1]]) * .25

    fx = convolve(im1, kernelX) + convolve(im2, kernelX)
    fy = convolve(im1, kernelY) + convolve(im2, kernelY)
    ft = convolve(im1, np.ones((2, 2)) * .25) + convolve(im2, -np.ones((2, 2)) * .25)
    return fx, fy, ft


def drawOpticalflow(img, U, V, step=7):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)

    flow = np.dstack((U, V))

    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.2)

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255,0))

    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def getUV(first_img, second_img, scale=3, threshold=4):
    U, V = HornSchunck(smoothen(first_img, 2), smoothen(second_img, 2))
    show_image = image_prev.copy()
    show_image = cv2.cvtColor(show_image, cv2.COLOR_BGR2GRAY)

    for i in range(0, len(U), 5):
        for j in range(0, len(V), 5):
            if np.absolute(U[i][j]) < threshold and np.absolute(V[i][j]) < threshold:
                U[i][j] = 0
                V[i][j] = 0

    cv2.imshow('Horn Schunck algorithm', drawOpticalflow(show_image, U * scale, V * scale))


def main():
    global image_prev
    image_prev = None

    # read the input file from cmd line
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        cap = cv2.VideoCapture(filename)
        while True:
            success, image = cap.read()
            if success:
                ch = cv2.waitKey(30)
                # press 'p' or 'P' to see the flow output
                if ch == (ord('p') or ord('P')):
                    # convert the image to grayscale
                    gray_prev = cv2.cvtColor(image_prev, cv2.COLOR_BGR2GRAY)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    getUV(gray_prev, gray)
                    while True:
                        if cv2.waitKey(1) & 0xFF == (ord('p') or ord('P')):
                            break
                elif ch == ord('q'):
                    break
                image_prev = image
                cv2.imshow('Horn Schunck algorithm', image)
            else:
                cv2.destroyAllWindows()
                cap.release()
                sys.exit()

        cv2.destroyAllWindows()
        cap.release()
        sys.exit()

    else:
        print("Please input a video file")
        sys.exit()

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()