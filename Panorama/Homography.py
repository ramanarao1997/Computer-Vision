import cv2
import numpy as np
import getopt
import sys
import random

# --NotebookApp.iopub_data_rate_limit=1.0e10

class Match:
    distance = 0
    img1_index = 0
    img2_index = 0

class Panorama:
    
    #  step 1
    def loadImage(self, filename):
        img = cv2.imread(filename, 0)
        if img is None:
            print('No image found:' + filename)
            return None
        else:
            print('Image loaded successfully!')

            return img

    #  step 2 and 3
    def getKeypointsandDescriptors(self,img):
        descriptors = cv2.xfeatures2d.SIFT_create()
        (keypoints, descriptors) = descriptors.detectAndCompute(img, None)

        return keypoints, descriptors

    #  step 4 and 5
    def matchFeatures(self, kp1, kp2, desc1, desc2, img1, img2, threshold):
        normalized_desc1 = []
        normalized_desc2 = []

        matches = []

        for des in desc1:
            tempDes = des / 99
            normalized_desc1.append(tempDes)

        for des in desc2:
            tempDes = des / 99
            normalized_desc2.append(tempDes)
            
        for i in range(len(desc1)):
            for j in range(len(desc2)):
                distance = np.linalg.norm(normalized_desc1[i] - normalized_desc2[j], ord = 2)
                
                if(distance<threshold):
                    match = Match()

                    match.distance = distance
                    match.img1_index = i
                    match.img2_index = j
                    
                    matches.append(match)
        
        print("\n Pairs of points whose distance is below a specific threshold:")

        for match in matches:
            print("[({0}, {1}), ({2}, {3})]".format(int(kp1[match.img1_index].pt[0]),
                    int(kp1[match.img1_index].pt[1]),
                    int(kp2[match.img2_index].pt[0]),
                    int(kp2[match.img2_index].pt[1])))

        return matches

    # step 6 helper
    def calculateHomography(self, correspondences):
        tempList = []
        for corr in correspondences:
            p1 = np.matrix([corr.item(0), corr.item(1), 1])
            p2 = np.matrix([corr.item(2), corr.item(3), 1])

            a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
                p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
            a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
                p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
            tempList.append(a1)
            tempList.append(a2)

        matrixA = np.matrix(tempList)

        u, s, v = np.linalg.svd(matrixA)

        h = np.reshape(v[8], (3, 3))

        h = (1/h.item(8)) * h

        return h

    # step 6 helper
    def geometricDistance(self,correspondence, h):

        p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
        estimatep2 = np.dot(h, p1)
        estimatep2 = (1/estimatep2.item(2))*estimatep2

        p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
        error = p2 - estimatep2
        return np.linalg.norm(error)

    # step 6
    def ransac(self,corr, threshold):
        maxInliers = []
        finalH = None
        avgResiduefinal = 0

        for i in range(10):
            corr1 = corr[random.randrange(0, len(corr))]
            corr2 = corr[random.randrange(0, len(corr))]
            randomFour = np.vstack((corr1, corr2))
            corr3 = corr[random.randrange(0, len(corr))]
            randomFour = np.vstack((randomFour, corr3))
            corr4 = corr[random.randrange(0, len(corr))]
            randomFour = np.vstack((randomFour, corr4))

            h = self.calculateHomography(randomFour)
            inliers = []
            avgResidue = 0

            for i in range(len(corr)):
                d = self.geometricDistance(corr[i], h)
                avgResidue = avgResidue + d

                if d < 5:
                    inliers.append(corr[i])

            avgResidue = avgResidue / len(corr)

            if len(inliers) > len(maxInliers):
                maxInliers = inliers
                finalH = h
                avgResiduefinal = avgResidue

            if len(maxInliers) > (len(corr)*threshold):
                break
        
        print("\n Average residual of inliers for the best fit  {0} ".format(avgResiduefinal))
            
        return finalH, maxInliers
    
    # step 6
    def drawMatches(self,img1, kp1, img2, kp2, matches, inliers = None):
        rows1 = img1.shape[0]
        cols1 = img1.shape[1]

        rows2 = img2.shape[0]
        cols2 = img2.shape[1]

        out = np.zeros((max([rows1,rows2]), cols1 + cols2, 3), dtype='uint8')
        out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])
        out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])
        
        for mat in matches:

            img1_idx = mat.img1_index
            img2_idx = mat.img2_index

            (x1,y1) = kp1[img1_idx].pt
            (x2,y2) = kp2[img2_idx].pt

            inlier = False

            if inliers is not None:
                for i in inliers:
                    if i.item(0) == x1 and i.item(1) == y1 and i.item(2) == x2 and i.item(3) == y2:
                        inlier = True

            cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
            cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

            if inliers is not None and inlier:
                cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 255, 0), 1)
            elif inliers is not None:
                cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 0, 255), 1)

            if inliers is None:
                cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

        return out

    # driver function
    def findHomographyUsingRANSAC(self, img1, img2):
        ransac_threshold = 1
        match_threshold = 0.5

        pointCorrespondences = []

        if img1 is not None and img2 is not None:

            # step - 2 & 3
            kp1, desc1 = self.getKeypointsandDescriptors(img1)
            kp2, desc2 = self.getKeypointsandDescriptors(img2)
            keypoints = [kp1, kp2]

            # step - 4 & 5
            pointMatches = self.matchFeatures(kp1, kp2, desc1, desc2, img1, img2, match_threshold)
            
            for match in pointMatches:
                (x1, y1) = keypoints[0][match.img1_index].pt
                (x2, y2) = keypoints[1][match.img2_index].pt
                pointCorrespondences.append([x1, y1, x2, y2])

            correspondences = np.matrix(pointCorrespondences)

            # step - 6
            homography, inliers = self.ransac(correspondences, ransac_threshold)
            
            print("\n Best fit homography: ")
            print(homography)

            print("\n Best fit inliers count: {0}".format(len(inliers)))

            matchedInliersImage = self.drawMatches(img1, kp1, img2, kp2, pointMatches, inliers)
            
        return homography, matchedInliersImage


    #  step 7 & 8
    def getPanorama(self, img1, img2, Homography):

        totalWidth = img1.shape[1] + img2.shape[1]
        panorama = cv2.warpPerspective(img1, Homography, (totalWidth , img2.shape[0]))
        panorama[0:img2.shape[0], 0:img2.shape[1]] = img2
        
        return panorama