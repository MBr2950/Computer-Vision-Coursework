import numpy as np
import cv2
import os

from scipy import stats

# Searches for an icon in an image
# Input: icon filepath; image filepath; threshold for accepting a match exists;
# threshold for rejecting outliers
# Output: array of matching points, each in the form [x, y]
def findMatchingPoints(icon, image, distanceThreshold):
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
    
    matchedImagePoints = cv2.KeyPoint_convert(matchedImagePoints)

    return matchedImagePoints

def findBoundingRectangles(image):
    # Tried using thresholding to solve problem of incorrect edge detection (doesn't identify grey edges),
    # neither this didn't seemed to help
    # returnValue, thresholdedImage = cv2.threshold(grey, 240, 255, cv2.THRESH_BINARY)

    # Conversion to greyscale seems to work better
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(grey, 100, 500)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    
    # Create a list of bounding boxes, round each continuous contour
    boundingRectangles = []
    maxContourArea = 0
    for contour in contours:
        rectangle = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > maxContourArea:
            maxContourArea = cv2.contourArea(contour)
        # Removes boxes created by noise in the data
        if cv2.contourArea(contour) >= 100:
            boundingRectangles.append(rectangle)
    print(maxContourArea)

    return boundingRectangles

def findIconInImage(icon, iconName, image, pointsThreshold, distanceThreshold, boundingRectangles):
    # Finds the matching points of the icon in the image, if they exist
    matchingPoints = findMatchingPoints(icon, image, distanceThreshold)

    bestMatch = 0
    bestMatchNumber = 0
    for i in range(len(boundingRectangles)):
        x, y, width, height = boundingRectangles[i]

        matchNumber = 0
        for point in matchingPoints:
            if point[0] >= x and point[0] <= x + width:
                if point[1] >= y and point[1] <= y + height:
                    matchNumber += 1

        if matchNumber > bestMatchNumber:
            bestMatchNumber = matchNumber
            bestMatch = i
        if matchNumber == bestMatchNumber:
            originalSize = boundingRectangles[bestMatch]
            originalSize = originalSize[2] * originalSize[3]
            newSize = width * height
            if newSize > originalSize:
                bestMatchNumber = matchNumber
                bestMatch = i

    if bestMatchNumber > pointsThreshold:
        x, y, width, height = boundingRectangles[bestMatch]
        return (bestMatch, bestMatchNumber, iconName)
    else:
        return None

def main():
    # Determines how many matching points are needed to count as a match, higher means stricter
    pointsThreshold = 5
    # Determines how similar two points have to be to count as a match, lower means stricter
    distanceThreshold = 1000

    # Icon is the smaller image to search for, image is the larger image to search in
    imagesLocation = './Task3Dataset/images/'
    imagePaths = os.listdir(imagesLocation)
    iconPaths = os.listdir('./IconDataset/png/')
    iconNames = ["Lighthouse", "Bike", "Bridge1", "Bridge", "Silo", "Church", "Supermarket", "Courthouse", "Airport", "Bench", "Bin", "Bus", "Water Well", "Flower",
                 "Barn", "House", "Cinema", "Bank", "Prison", "ATM", "Solar Panel", "Car", "Traffic Light", "Fountain", "Factory", "Shop", "Petrol Station", "Government",
                 "Theatre", "Telephone Box", "Field", "Van", "Hydrant", "Billboard", "Police Station", "Hotel", "Post Office", "Library", "University", "Bus Stop", "Windmill",
                 "Tractor", "Sign", "Ferris Wheel", "Museum", "Fire Station", "Restaurant", "Hospital", "School", "Cemetery"]
    
    icons = []
    for iconPath in iconPaths:
        path = './IconDataset/png/' + iconPath
        icon = cv2.imread(path)
        # Slight blurring seems to remove false positives
        # icon = cv2.blur(icon, (2, 2))
        icons.append(icon)

    for i in range(len(imagePaths)):
        imagePath = imagesLocation + imagePaths[i]
        image = cv2.imread(imagePath)

        boundingRectangles = findBoundingRectangles(image)

        # Limits each bounding box to at most one matching icon, using the closest one, to limit false positives
        bestMatchInfo = []
        for j in range(len(iconPaths)):
            bestMatch = findIconInImage(icons[j], iconNames[j], image, pointsThreshold, distanceThreshold, boundingRectangles)
            if bestMatch != None:
                bestMatchInfo.append(bestMatch)

        for j in range(len(boundingRectangles)):
            bestMatch = 0
            bestMatchNumber = 0

            x, y, width, height = boundingRectangles[j]

            for k in range(len(bestMatchInfo)):
                if bestMatchInfo[k][0] == j:
                    if bestMatchInfo[k][1] > bestMatchNumber:
                        bestMatch = k
                        bestMatchNumber = bestMatchInfo[k][1]

            if bestMatchNumber > 0:
                x, y, width, height = boundingRectangles[bestMatchInfo[bestMatch][0]]
                iconName = bestMatchInfo[bestMatch][2]
                print(f"Icon: {iconName}, Position (x, y, width, height): {x, y, width, height}")
                cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(image, iconName, (x + width + 10,y + height), 0, 0.3, (0, 255, 0))


        cv2.imshow('image', cv2.resize(image, (800, 600)))
        k = cv2.waitKey(0) & 0xff

        if type(image) == np.ndarray:
            # cv2.imshow('image', cv2.resize(image, (800, 600)))

            # k = cv2.waitKey(0) & 0xff

            # if k == 27:
            #     cv2.destroyAllWindows()

            cv2.imwrite("./Task3OutputImages/" + imagePaths[i], image)
            print("Image", i + 1, "done")
    

main()