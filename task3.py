import numpy as np
import cv2
import os

from scipy import stats

# Searches for an icon in an image
# Input: icon filepath; image filepath; threshold for accepting a match exists;
# threshold for rejecting outliers
# Output: array of matching points, each in the form [x, y]
def findMatchingPoints(icon, image, distanceThreshold, zScoreThreshold):
    icon = cv2.cvtColor(icon, cv2.COLOR_RGB2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()

    # each descriptor is an array, kinda like a hash of the point 
    keypoints1, descriptors1 = sift.detectAndCompute(icon, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image, None)
    # print(cv2.KeyPoint_convert(keypoints2))

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key = lambda x: x.distance)

    # Add all keypoints in the test image that match a keypoint in the icon to array
    matchedImagePoints = []
    for match in matches:
        if match.distance < distanceThreshold:
            matchedImagePoints.append(keypoints2[match.trainIdx])
            # print(match.imgIdx, match.queryIdx, match.trainIdx)
    
    # Removes outlier points to narrow down icon location
    matchedImagePoints = cv2.KeyPoint_convert(matchedImagePoints)
    matchedImagePoints = removeOutliers(matchedImagePoints, zScoreThreshold)

    return matchedImagePoints

# Takes in a numpy array, returns numpy array with outliers removed
# Input: initial array; threshold for outlier detection
# Output: array with outliers removed
def removeOutliers(data, threshold):
    # zscore gives a measure of how far an element is from the mean
    zScores = np.abs(stats.zscore(data))

    # Removes all elements with the zscore for x or y greater than the threshold
    cleanedData = []
    for i in range(len(data)):
        # Maybe change this to combination of zscore for x and y?
        if zScores[i][0] <= threshold and zScores[i][1] <= threshold:
            cleanedData.append(data[i])

    return cleanedData

def findBoundingSquare(matchingPoints):
    leftmostPoint = 1000
    rightmostPoint = 0
    bottommostPoint = 1000
    topmostPoint = 0

    for point in matchingPoints:
        if point[0] < leftmostPoint:
            leftmostPoint = point[0]
        elif point[0] > rightmostPoint:
            rightmostPoint = point[0]

        if point[1] < bottommostPoint:
            bottommostPoint = point[1]
        elif point[1] > topmostPoint:
            topmostPoint = point[1]

    return ((int(leftmostPoint), int(bottommostPoint)), (int(rightmostPoint), int(topmostPoint)))

def findIconInImage(icon, iconName, image, pointsThreshold):
    originalImage = image

    # Finds the matching points of the icon in the image, if they exist
    matchingPoints = findMatchingPoints(icon, image, 1250, 2)

    # Finds the centre of these points, if there are enough to constitute a match
    if len(matchingPoints) >= pointsThreshold:
        centre = np.mean(matchingPoints, axis = 0)
    else:
        centre = None

    # If the icon does exist in the image, draw a circle where the detected centre is
    if type(centre) == np.ndarray:
        # Add dot at centre of icon
        matchesImage = cv2.circle(originalImage, (int(centre[0]), int(centre[1])), radius=5, color=(255, 0, 255), thickness=-1)

        boundingPoints = findBoundingSquare(matchingPoints)
        matchesImage = cv2.rectangle(matchesImage, boundingPoints[0], boundingPoints[1], color = (0, 0, 255))
        # print(boundingPoints)
        textPosition = (boundingPoints[0][0], boundingPoints[1][1] + 15)
        font = cv2.FONT_HERSHEY_SIMPLEX
        matchesImage = cv2.putText(matchesImage, iconName, textPosition, font, 0.5, (0, 0, 255))

        return matchesImage
    # Otherwise it does not exist in the image
    else:
        # print("No Match")
        return originalImage

def main():
    # Determines how many matching points are needed to count as a match
    pointsThreshold = 10

    # Icon is the smaller image to search for, image is the larger image to search in
    imagePaths = os.listdir('./Task3Dataset/images/')
    iconPaths = os.listdir('./IconDataset/png/')
    iconNames = ["Lighthouse", "Bike", "Bridge1", "Bridge", "Silo", "Church", "Supermarket", "Courthouse", "Airport", "Bench", "Bin", "Bus", "Water Well", "Flower",
                 "Barn", "House", "Cinema", "Bank", "Prison", "ATM", "Solar Panel", "Car", "Traffic Light", "Fountain", "Factory", "Shop", "Petrol Station", "Government",
                 "Theatre", "Telephone Box", "Field", "Van", "Hydrant", "Billboard", "Police Station", "Hotel", "Post Office", "Library", "University", "Bus Stop", "Windmill",
                 "Tractor", "Sign", "Ferris Wheel", "Museum", "Fire Station", "Restaurant", "Hospital", "School", "Cemetery"]

    for i in range(len(imagePaths)):
        imagePath = './Task3Dataset/images/' + imagePaths[i]
        image = cv2.imread(imagePath)

        for j in range(len(iconPaths)):
            iconPath = './IconDataset/png/' + iconPaths[j]
            icon = cv2.imread(iconPath)
    
            image = findIconInImage(icon, iconNames[j], image, pointsThreshold)

        if type(image) == np.ndarray:
            # cv2.imshow('image', cv2.resize(image, (800, 600)))

            # k = cv2.waitKey(0) & 0xff

            # if k == 27:
            #     cv2.destroyAllWindows()

            cv2.imwrite("./Task3OutputImages/" + imagePaths[i], image)
    

main()