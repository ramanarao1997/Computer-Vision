import numpy as np
import cv2
import sys
import video_23 as video

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, False, (0, 255,0), 1)
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        cv2.circle(vis, (x2,y2), 1, (0,255,0),-1)
    return vis

if __name__ == '__main__':

    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0   

    # Parameters for Lucas Kanade optical flow
    cap = video.create_capture(fn)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take the first frame from the video and convert it to grayscale
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    height = old_gray.shape[0]
    width = old_gray.shape[1]

    while True:
        ret, frame = cap.read()

        if frame is None:
            break
        # convert the current frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p = []
        for i in range(0,frame_gray.shape[0]):
            for j in range(0,frame_gray.shape[1]):
                p.append([[np.float32(i), np.float32(j)]])

        all_pixels = np.asarray(p, dtype = np.float32)
        all_pixels = np.reshape(all_pixels,(-1,1,2))

        # Calculate Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, all_pixels, None, **lk_params)
        all_pixels = all_pixels.reshape(height, width, 2)
        p1 = p1.reshape(height, width, 2)

        # calculate the flow vector by subtraction
        flow = p1 - all_pixels

        # update the old frame as current frame
        old_gray = frame_gray.copy()

        # visualize the result
        vis = draw_flow(frame_gray, flow)
        cv2.imshow('Dense Lucas Kanade',vis)

        # press 'q' key to exit
        ch = cv2.waitKey(1) & 0xFF
        if ch == ord('q'):
            break

    cv2.destroyAllWindows()