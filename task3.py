import numpy as np
import cv2
import os
import time
import csv


# Finds the keypoints in each image and icon
# Input: list of all images as cv2 images; list of all icons as cv2 images
# Output: list of keypoint descriptors in image; list of keypoint locations in image; list of keypoint descriptors in icon
def findKeyPoints(icons, images):
    # Each descriptor is an array, kind of like a hash of the point
    iconsDescriptors = []
    imagesDescriptors = []
    imagesKeypoints = []

    # Uses SIFT to identify keypoints in each icon and image
    sift = cv2.SIFT_create()
    for icon in icons:
        iconKeypoints, iconDescriptors = sift.detectAndCompute(icon, None)
        iconsDescriptors.append(iconDescriptors)
    for image in images:
        imageKeypoints, imageDescriptors = sift.detectAndCompute(image, None)
        imagesKeypoints.append(imageKeypoints)
        imagesDescriptors.append(imageDescriptors)

    # Icon keypoint location is unnecessary
    return imagesDescriptors, imagesKeypoints, iconsDescriptors


# Generates bounding rectangles around all closed shapes in an image
# Input: image as cv2 image
# Output: tuple of bounding rectangles, all in the format [x, y, width, height]
def findBoundingRectangles(image):
    # Thresholding makes it easier to detect grey edges, need to convert to greyscale to do this
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    returnValue, grey = cv2.threshold(grey, 240, 255, cv2.THRESH_BINARY_INV)

    # Detects edges, then converts this to a contour map
    edges = findEdges(grey)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a list of bounding boxes, round each continuous contour
    boundingRectangles = []
    for contour in contours:
        rectangle = cv2.boundingRect(contour)
        # Removes small boxes created by noise in the data
        if rectangle[2] * rectangle[3] >= 100:
            boundingRectangles.append(rectangle)

    return boundingRectangles


# Use canny edge detection to identify edge in image
# Input: Image in the form of a numpy array
# Output: Edgemap in the form of a numpy array
def findEdges(image : np.ndarray, highFraction : float = 0.1, lowFraction : float = 0.01):
    # Find gradient
    sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Find magnitude of gradient
    intensity = np.sqrt(sobelX**2 + sobelY**2)
    intensity = intensity / np.max(intensity) * 255
    edges = intensity.astype(np.uint8)

    # Find direction of gradient
    direction = np.arctan2(sobelY, sobelX)

    # # NON-MAX SUPPRESSION
    # This is removed, as using it results in non-continuous edges, which are needed to
    # create accurate contours and bounding boxes
    # supressionMat = np.zeros_like(edges)

    # for y in range(1, edges.shape[0] - 1):
    #     for x in range(1, edges.shape[1] - 1):
            
    #         f = 255
    #         b = 255

    #         angle = direction[y, x] * 180 / np.pi
    #         angle = abs(angle)

    #         if (angle >= 0 and angle < 22.5) or (angle >= 157.5 and angle <= 180):
    #             f = edges[y - 1, x + 1]
    #             b = edges[y + 1, x - 1]
    #         elif (angle >= 22.5 and angle < 67.5):
    #             f = edges[y - 1, x]
    #             b = edges[y + 1, x]
    #         elif (angle >= 67.5 and angle < 112.5):
    #             f = edges[y - 1, x - 1]
    #             b = edges[y + 1, x + 1]
    #         elif (angle >= 112.5 and angle < 157.5):
    #             f = edges[y, x - 1]
    #             b = edges[y, x + 1]
           
           
    #         if (edges[y, x] >= f) and (edges[y, x] >= b):
    #             supressionMat[y, x] = edges[y, x]

    # edges = supressionMat

    # DOUBLE THRESHOLD
    highThreshold = edges.max() * highFraction
    lowThreshold = edges.max() * lowFraction

    thresholdMat = np.zeros_like(edges)

    strongI = np.where(edges > highThreshold)
    weakI = np.where((edges <= highThreshold) & (edges >= lowThreshold))

    thresholdMat[strongI] = 255
    thresholdMat[weakI] = 45

    edges = thresholdMat

    
    # HYSTERESIS
    for y in range(1, edges.shape[0] - 1):
        for x in range(1, edges.shape[1] - 1):
            if (edges[y, x] == 50):
                if (edges[y - 1, x - 1] == 255) or (edges[y - 1, x] == 255) or \
                (edges[y - 1, x + 1] == 255) or (edges[y, x - 1] == 255) or \
                (edges[y, x + 1] == 255) or (edges[y + 1, x - 1] == 255) or \
                (edges[y + 1, x] == 255) or (edges[y + 1, x + 1] == 255):
                    edges[y, x] = 255
                else:
                    edges[y, x] = 0

    return edges


# Given one icon, tries to find if it exists in one image
# Inputs: threshold for determining if two points match; threshold for determining how many points to count a match; 
# icon's name, for labelling; list of keypoint descriptors in icon; list of keypoint descriptors in image; 
# list of keypoint locations in image; list of bounding rectangles of shapes in image
# Output: tuple of information about the bounding box most likely to contain the icon, contains index of best match,
# how many matching points in best matching icon and box, and icon's name
def findIconInImage(distanceThreshold, pointsThreshold, iconName, iconDescriptors, imageDescriptors, imageKeypoints, boundingRectangles, losstype):
    # Finds the matching points of the icon in the image, if they exist
    matchingPoints = findMatchingPoints(distanceThreshold, iconDescriptors, imageDescriptors, imageKeypoints, losstype)

    # bestMatch is the index of the bounding box most likely to contain icon
    bestMatch = 0
    # bestMatchMagnitude is how many of the icon's points are in this box
    bestMatchMagnitude = 0
    for i in range(len(boundingRectangles)):
        x, y, width, height = boundingRectangles[i]

        # Check how many icon points are in each bounding box
        matchNumber = 0
        for point in matchingPoints:
            if point[0] >= x and point[0] <= x + width:
                if point[1] >= y and point[1] <= y + height:
                    matchNumber += 1

        # Only keep best matching box
        if matchNumber > bestMatchMagnitude:
            bestMatchMagnitude = matchNumber
            bestMatch = i

    # If there are enough points in any box to constitute a match, return information about that box 
    if bestMatchMagnitude >= pointsThreshold:
        x, y, width, height = boundingRectangles[bestMatch]

        return (bestMatch, bestMatchMagnitude, iconName)
    else:
        return None


# Searches for an icon in an image
# Input: threshold for accepting a match exists; keypoint descriptors for icon;
# keypoint descriptors for image; keypoint locations for image; loss type used
# Output: array of matching points, each in the form [x, y]
def findMatchingPoints(distanceThreshold, iconDescriptors, imageDescriptors, imageKeypoints, losstype):
    # Uses sift to generate list of matching points between icon and image
    matches = matchPoints(iconDescriptors, imageDescriptors, losstype)
    # Sorts the matches in ascending order, starting with the closest match 
    matches = sorted(matches, key = lambda x: x.distance)

    # Add all keypoints in the test image that match a keypoint in the icon to array
    matchedImagePoints = []
    for match in matches:
        if match.distance < distanceThreshold:
            matchedImagePoints.append(imageKeypoints[match.trainIdx])
            # print(match.imgIdx, match.queryIdx, match.trainIdx)
    
    # Returns matched points in an easier to use format
    matchedImagePoints = cv2.KeyPoint_convert(matchedImagePoints)
    return matchedImagePoints


# Matches points between two sets of descriptors
# Input: first list of descriptors; second list of descriptors; loss type used
# Output: list of matching points, of the type cv2.DMatch
def matchPoints(descriptors1, descriptors2, losstype):
    matches = []
    # Loop through each descriptor in the first set
    for i, descriptor1 in enumerate(descriptors1):
        bestMatch = None
        bestDistance = float('inf')
        # Loop through each descriptor in the second set
        for j, descriptor2 in enumerate(descriptors2):
            # Compute distance between descriptors
            distance = np.linalg.norm(descriptor1 - descriptor2, ord = losstype)
            # Update best match if distance is smaller
            if distance < bestDistance:
                bestMatch = j
                bestDistance = distance
                
        # Compute distance from descriptor2 to descriptor1
        reverseDistance = np.linalg.norm(descriptors2[bestMatch] - descriptor1, ord = losstype)
        # If the match is not symmetric, continue to the next descriptor in set 1
        if reverseDistance > bestDistance:
            continue
        
        if bestMatch != None:
            # Append the best match to the list of matches
            matches.append(cv2.DMatch(i, bestMatch, bestDistance))
    return matches


# Calculates the 'IntersectionOverUnion' metric, a measure of what percentage of the bounding box
# identified lines up with the actual bounding box
# Inputs: First box, in the format (x1, y1, x2, y2); second box, in the format (x1, y1, x2, y2) 
def calculateIntersectionOverUnion(box1, box2):
    # 0 are top left coordinates, 1 are bottom right coordinates
    XA0, YA0, XA1, YA1 = box1
    XB0, YB0, XB1, YB1 = box2

    # Finds the coordinates of the overlap between the boxes
    X0Inter = max(XA0, XB0)
    Y0Inter = max(YA0, YB0)
    X1Inter = max(XA1, XB1)
    Y1Inter = max(YA1, YB1)

    # If the width of the height are negative, the boxes don't overlap at all
    intersectionWidth = (X1Inter - X0Inter)
    intersectionHeight = (Y1Inter - Y0Inter)
    
    # Calculate overlap (intersection over union)
    if intersectionWidth >= 0 and intersectionHeight >= 0: 
        intersectionArea = intersectionWidth * intersectionHeight
        unionArea = (XA1 - XA0) * (YA1 - YA0) + (XB1 - XB0) * (YB1 - YB0) - intersectionArea

        intersectionOverUnion = intersectionArea / unionArea
    else:
        intersectionOverUnion = 0

    return intersectionOverUnion


# Identifies icons within images, saves images with bounding boxes around identified icons
# Default values are those that have been found to give best accuracy
# Inputs: threshold for determining if two points match; threshold for determining how many points to count a match;
# loss type used; path to folder containing icons; path to folder containing images; path to folder containing image annotations; 
# path to folder to output images with bounding boxes
# Output: number of true positives, number of false positives, number of false negatives, average intersection over union, 
# total runtime
def main(distanceThreshold = 60, pointsThreshold = 1, losstype = cv2.NORM_L2, iconsLocation = './IconDataset/png/', 
         imagesLocation = './Task3Dataset/images/', annotationsLocation = './Task3Dataset/annotations/',
         outputLocation = "./Task3Dataset/output-images/"):
    startTime = time.time()

    # Icon is the smaller image to search for, image is the larger image to search in
    iconPaths = os.listdir(iconsLocation)
    # List of icon names for labelling/checking against annotations
    iconNames = ["lighthouse", "bike", "bridge-1", "bridge", "silo", "church", "supermarket", "courthouse", "airport", "bench", "trash", "bus", "water-well", "flower",
                 "barn", "house", "cinema", "bank", "prison", "atm", "solar-panel", "car", "traffic-light", "fountain", "factory", "shop", "gas-station", "government",
                 "theater", "telephone-booth", "field", "van", "hydrant", "billboard", "police", "hotel", "post-office", "library", "university", "bus-stop", "windmill",
                 "tractor", "sign", "ferris-wheel", "museum", "fire-station", "restaurant", "hospital", "school", "cemetery"]
    imagePaths = os.listdir(imagesLocation)
    annotationPaths = os.listdir(annotationsLocation)
    
    # Reads all icons and images to cv2 images
    icons = []
    images = []

    for iconPath in iconPaths:
        path = iconsLocation + iconPath
        icon = cv2.imread(path)
        icons.append(icon)
    for imagePath in imagePaths:
        imagePath = imagesLocation + imagePath
        image = cv2.imread(imagePath)
        images.append(image)

    # Identifies keypoint info in images and icons
    imagesDescriptors, imagesKeypoints, iconsDescriptors = findKeyPoints(icons, images)

    keypointsTime = time.time()
    print("Time after identifying keypoints:", keypointsTime - startTime)

    # Used for measuring accuracy
    truePositives = 0
    falsePositives = 0
    falseNegatives = 0
    numIcons = 0
    intersectionOverUnions = 0

    # Finds icons in each image
    for i in range(len(images)):
        # Get information about image
        annotations = []
        file = open(annotationsLocation + annotationPaths[i], mode ='r')
        lines = csv.reader(file)
        for line in lines:
            annotations.append(line)
        annotations = annotations[1:]
        numIcons += len(annotations)
        file.close()

        # Finds all bounding rectangles within each image
        boundingRectangles = findBoundingRectangles(images[i])

        # Finds the bounding box the icon is most likely to fit in
        bestMatchInfo = []
        for j in range(len(iconPaths)):
            bestMatch = findIconInImage(distanceThreshold, pointsThreshold, iconNames[j], iconsDescriptors[j], 
                                        imagesDescriptors[i], imagesKeypoints[i], boundingRectangles, losstype)
            if bestMatch != None:
                bestMatchInfo.append(bestMatch)

        # Limits each bounding box to at most one matching icon, using the closest one, to limit false positives
        for j in range(len(boundingRectangles)):
            bestMatch = 0
            bestMatchMagnitude = 0

            x, y, width, height = boundingRectangles[j]

            for k in range(len(bestMatchInfo)):
                match, matchMagnitude, _ = bestMatchInfo[k]
                if match == j:
                    if matchMagnitude > bestMatchMagnitude:
                        bestMatch = k
                        bestMatchMagnitude = matchMagnitude

            # If a match does exist, draws the best one onto the image, and labels it
            if bestMatchMagnitude > 0:
                x, y, width, height = boundingRectangles[bestMatchInfo[bestMatch][0]]
                iconName = bestMatchInfo[bestMatch][2]
                print(f"Icon: {iconName}, Position (x, y, width, height): {x, y, width, height}, Number of Matching Points: {bestMatchMagnitude}")
                cv2.rectangle(images[i], (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(images[i], iconName, (x + width + 10, y + height), 0, 0.3, (0, 255, 0))

                # Checks if the match is correct, to add to true positives or false positives
                truePositive = False
                for annotation in annotations:
                    if iconName == annotation[0]:
                        truePositives += 1
                        truePositive = True

                        x2 = x + width
                        y2 = y + height
                        intersectionOverUnions += calculateIntersectionOverUnion((x, y, x2, y2), (int(annotation[1]), int(annotation[2]), int(annotation[3]), int(annotation[4])))

                if truePositive == False:
                    falsePositives += 1

        # Finds number of icons not detected, and average intersectionOverUnion (including false negatives,
        # on average how much do predicted bounding boxes line up with actual bounding boxes)
        falseNegatives = numIcons - truePositives
        averageIntersectionOverUnion = (intersectionOverUnions / numIcons) * 100

        # Saves the image, if is of right type
        if type(images[i]) == np.ndarray:
            cv2.imwrite(outputLocation + imagePaths[i], images[i])
            print("Image", i + 1, "done")

    finalTime = time.time() - startTime
    print("End Time:", finalTime)

    print(f"True Positives: {truePositives}, False Positives: {falsePositives}, False Negatives: {falseNegatives}, IoU: {averageIntersectionOverUnion}")
    return truePositives, falsePositives, falseNegatives, averageIntersectionOverUnion, finalTime

# Test the code for a number of different parameters
if __name__ == '__main__':
    accuracies = []
    for pointsThreshold in range(1, 4):
        for distanceThreshold in range(0, 101, 5):
            print("\n")
            print(f"Distance Threshold: {distanceThreshold}, Points Threshold: {pointsThreshold}")
            TPs, FPs, FNs, intersectionOverUnion, finalTime = main(distanceThreshold, pointsThreshold)

            accuracy = TPs / (TPs + FPs + FNs)
            accuracy *= 100

            TPR = TPs / (TPs + FNs)
            TPR *= 100

            accuracies.append((distanceThreshold, pointsThreshold, TPs, FPs, FNs, accuracy, TPR, intersectionOverUnion, finalTime))

            print("\n")
            for line in accuracies:
                print(f"Distance Threshold: {line[0]}, Points Threshold: {line[1]}, TPs: {line[2]}, FPs: {line[3]}, FNs: {line[4]}, Accuracy: {line[5]}%, TPR: {line[6]}%, IOU: {line[7]}%")