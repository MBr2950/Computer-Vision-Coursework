# TODO: Use minAreaRect to find orientation, add parallelism

import numpy as np
import cv2
import os
import time

# Searches for an icon in an image
# Input: icon cv2 image; image cv2 image; threshold for accepting a match exists;
# keypoint descriptors for image; keypoint locations for image; keypoint descriptors for icon
# Output: array of matching points, each in the form [x, y]
def findMatchingPoints(distanceThreshold, imageDescriptors, imageKeypoints, iconDescriptors):
    # Uses sift to generate list of matching points between icon and image
    matches = matchPoints(iconDescriptors, imageDescriptors)
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

def matchPoints(descriptors1, descriptors2):
    matches = []
    # Loop through each descriptor in the first set
    for i, desc1 in enumerate(descriptors1):
        best_match = None
        best_distance = float('inf')
        # Loop through each descriptor in the second set
        for j, desc2 in enumerate(descriptors2):
            # Compute distance between descriptors
            distance = np.linalg.norm(desc1 - desc2, ord=cv2.NORM_L2)
            # Update best match if distance is smaller
            if distance < best_distance:
                best_match = j
                best_distance = distance
                
        # Compute distance from desc2 to desc1
        distance_reverse = np.linalg.norm(descriptors2[best_match] - desc1, ord=cv2.NORM_L2)
        # If the match is not symmetric, continue to the next descriptor in set 1
        if distance_reverse > best_distance:
            continue

        # Append the best match to the list of matches
        matches.append(cv2.DMatch(i, best_match, best_distance))
    return matches

# Generates bounding rectangles around all closed shapes in an image
# Input: image as cv2 image
# Output: tuple of bounding rectangles, all in the format [x, y, width, height]
def findBoundingRectangles(image):
    # Conversion to greyscale seems to work better
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Tried using thresholding to solve problem of incorrect edge detection (doesn't identify grey edges),
    # neither this didn't seemed to help
    # returnValue, grey = cv2.threshold(grey, 240, 255, cv2.THRESH_BINARY)

    # Detects edges, then converts this to a contour map
    edges = cv2.Canny(grey, 100, 200)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    
    
    # Create a list of bounding boxes, round each continuous contour
    boundingRectangles = []
    # maxContourArea = 0
    for contour in contours:
        rectangle = cv2.boundingRect(contour)
        # if cv2.contourArea(contour) > maxContourArea:
        #     maxContourArea = cv2.contourArea(contour)
        # Removes small boxes created by noise in the data
        if cv2.contourArea(contour) >= 100:
            boundingRectangles.append(rectangle)
            # cv2.rectangle(image, (rectangle[0], rectangle[1]), (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]), (0, 255, 0))
    # print(maxContourArea)
    
    # cv2.imshow('image', cv2.resize(image, (800, 600)))
    # k = cv2.waitKey(0) & 0xff

    return boundingRectangles

# Given one icon, tries to find if it exists in one image
# Inputs: icon as a cv2 image; icon's name, for labelling; threshold for determining how many points to count a match;
# threshold for determining if two points match; list of bounding rectangles of shapes in image; list of keypoint descriptors in image;
# list of keypoint locations in image; list of keypoint descriptors in icon
# Output: tuple of information about the bounding box most likely to contain the icon
def findIconInImage(iconName,pointsThreshold, distanceThreshold, boundingRectangles, imageDescriptors, imageKeypoints, iconDescriptors):
    # Finds the matching points of the icon in the image, if they exist
    matchingPoints = findMatchingPoints(distanceThreshold, imageDescriptors, imageKeypoints, iconDescriptors)

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

# Finds the keypoints in each image and icon
# Input: list of all images as cv2 images; list of all icons as cv2 images
# Output: list of keypoint descriptors in image; list of keypoint locations in image; list of keypoint descriptors in icon
def findKeyPoints(images, icons):
    # Each descriptor is an array, kind of like a hash of the point 
    imagesDescriptors = []
    imagesKeypoints = []
    iconsDescriptors = []

    # Uses SIFT to identify keypoints in each, converting to greyscale seems to help performance
    sift = cv2.SIFT_create()
    for image in images:
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        imageKeypoints, imageDescriptors = sift.detectAndCompute(image, None)
        imagesKeypoints.append(imageKeypoints)
        imagesDescriptors.append(imageDescriptors)
    for icon in icons:
        # icon = cv2.cvtColor(icon, cv2.COLOR_RGB2GRAY)
        iconKeypoints, iconDescriptors = sift.detectAndCompute(icon, None)
        iconsDescriptors.append(iconDescriptors)

    # Icon keypoint location is unnecessary
    return imagesDescriptors, imagesKeypoints, iconsDescriptors
        
# Identifies icons within images, saves images with bounding boxes around identified icons
# Input: None
# Output: None
def main():
    startTime = time.time()

    # Determines how many matching points are needed to count as a match, higher means stricter
    # For unrotated: 5, for rotated: 3
    pointsThreshold = 5
    # Determines how similar two points have to be to count as a match, lower means stricter
    # For unrotated: 50, for rotated: 1000
    distanceThreshold = 50

    # Icon is the smaller image to search for, image is the larger image to search in
    imagesLocation = './Task3Dataset/images/'
    imagePaths = os.listdir(imagesLocation)
    iconPaths = os.listdir('./IconDataset/png/')
    iconNames = ["Lighthouse", "Bike", "Bridge1", "Bridge", "Silo", "Church", "Supermarket", "Courthouse", "Airport", "Bench", "Bin", "Bus", "Water Well", "Flower",
                 "Barn", "House", "Cinema", "Bank", "Prison", "ATM", "Solar Panel", "Car", "Traffic Light", "Fountain", "Factory", "Shop", "Petrol Station", "Government",
                 "Theatre", "Telephone Box", "Field", "Van", "Hydrant", "Billboard", "Police Station", "Hotel", "Post Office", "Library", "University", "Bus Stop", "Windmill",
                 "Tractor", "Sign", "Ferris Wheel", "Museum", "Fire Station", "Restaurant", "Hospital", "School", "Cemetery"]
    
    
    # Reads all icons and images to cv2 images
    images = []
    icons = []
    for imagePath in imagePaths:
        imagePath = imagesLocation + imagePath
        image = cv2.imread(imagePath)
        images.append(image)
    for iconPath in iconPaths:
        path = './IconDataset/png/' + iconPath
        icon = cv2.imread(path)
        icons.append(icon)

    # Identifies keypoint info in images and icons
    imagesDescriptors, imagesKeypoints, iconsDescriptors = findKeyPoints(images, icons)

    keypointsTime = time.time()
    print("Time after identifying keypoints:", keypointsTime - startTime)

    for i in range(len(images)):
        # Finds all bounding rectangles within each image
        boundingRectangles = findBoundingRectangles(images[i])

        # Finds the bounding box the icon is most likely to fit in
        bestMatchInfo = []
        for j in range(len(iconPaths)):
            bestMatch = findIconInImage(iconNames[j], pointsThreshold, distanceThreshold, boundingRectangles,
                                         imagesDescriptors[i], imagesKeypoints[i], iconsDescriptors[j])
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

            # If a match does exist, draws the best on onto the image, and labels it
            if bestMatchMagnitude > 0:
                x, y, width, height = boundingRectangles[bestMatchInfo[bestMatch][0]]
                iconName = bestMatchInfo[bestMatch][2]
                print(f"Icon: {iconName}, Position (x, y, width, height): {x, y, width, height}, Number of Matches: {bestMatchMagnitude}")
                cv2.rectangle(images[i], (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(images[i], iconName, (x + width + 10,y + height), 0, 0.3, (0, 255, 0))


        # cv2.imshow('image', cv2.resize(images[i], (800, 600)))
        # k = cv2.waitKey(0) & 0xff

        # Saves the image, if is of right type
        if type(images[i]) == np.ndarray:
            cv2.imwrite("./Task3OutputImages/" + imagePaths[i], images[i])
            print("Image", i + 1, "done")
    
    finalTime = time.time()
    print("End Time:", finalTime - startTime)

main()