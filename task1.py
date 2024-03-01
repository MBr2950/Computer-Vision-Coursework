import cv2
import numpy as np
from hough import houghTransform, findMaxima

def main():
    for i in range(1, 11):
        image = cv2.imread("./Task1Dataset/image" + str(i) + ".png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # cv2.imshow("image", image)
        # cv2.waitKey(1000)

        edges = findEdges(image)
        lines = findLines(edges)
        points = findPoints(lines)
        print(points)
        angle = findAngle(points)
        print(angle)

# Find the angle specified by the 3 points identified
# Input: List of 3 lists, each of format [x, y] and representing either the end of a line, or the intersection point
# Output: Angle in degrees
def findAngle(points):
    distanceA = findDistance([points[0], points[2]])
    distanceB = findDistance([points[1], points[2]])
    distanceC = findDistance([points[0], points[1]])

    angle = np.arccos((distanceA**2 + distanceB**2 - distanceC**2) / (2 * distanceA * distanceB))
    angle = angle * (180 / np.pi)

    return angle

# Find the distance between 2 points
# Input: List of 2 lists, each of format [x, y] and representing a point
# Output: Euclidean distance as float
def findDistance(points):
    dx = points[1][0] - points[0][0]
    dy = points[1][1] - points[0][1]
    distance = np.sqrt(dx**2 + dy**2)

    return distance

# Use canny edge detection to identify edge in image
# Input: Image in the form of a numpy array
# Output: Edgemap in the form of a numpy array
def findEdges(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

# Convert edgemap into lines
# Input: Edgemap in the form of a numpy array
# Output: Detected lines, each represented by start and end coordinates in the format [[x1, y1, x2, y2]]
def findLines(edges):
    rho = 1
    theta = np.pi / 180
    threshold = 8
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=98, maxLineGap=48)

    return lines

# Try to identify the 3 points of the angle (end points and intersection)
# Input: Detected lines, each represented by start and end coordinates in the format [[x1, y1, x2, y2]]
# Output: List of 3 lists, each of format [x, y] and representing either the end of a line, or the intersection point
def findPoints(lines):
    tolerance = 10
    # Choose points for basis of final points
    points = [[lines[0][0][0], lines[0][0][1]], [lines[0][0][2], lines[0][0][3]], None]
    # Uses a voting system to try and determine points
    counters = [0, 0, 0]
    
    for line in lines:
        # Named variable for easier reading
        xStart = line[0][0]
        xEnd = line[0][2]
        yStart = line[0][1]
        yEnd = line[0][3]

        start = False
        end = False

        # If the start or end coordinates in a line are similar to an existing point, add vote for this point
        # and move slightly
        if abs(xStart - points[0][0]) < tolerance and abs(yStart - points[0][1]) < tolerance:
            points[0][0] = (points[0][0] + xStart) / 2
            points[0][1] = (points[0][1] + yStart) / 2
            start = True
            counters[0] += 1

        if abs(xEnd - points[0][0]) < tolerance and abs(yEnd - points[0][1]) < tolerance:
            points[0][0] = (points[0][0] + xEnd) / 2
            points[0][1] = (points[0][1] + yEnd) / 2
            end = True
            counters[0] += 1

        if abs(xStart - points[1][0]) < tolerance and abs(yStart - points[1][1]) < tolerance:
            points[1][0] = (points[1][0] + xStart) / 2
            points[1][1] = (points[1][1] + yStart) / 2
            start = True
            counters[1] += 1

        if abs(xEnd - points[1][0]) < tolerance and abs(yEnd - points[1][1]) < tolerance:
            points[1][0] = (points[1][0] + xEnd) / 2
            points[1][1] = (points[1][1] + yEnd) / 2
            end = True
            counters[1] += 1

        # If points[2] hasn't been instantiated, and this is a valid point, set points[2] as the other end of the line 
        if points[2] == None:
            if start == True and end == False:
                points[2] = [xEnd, yEnd]
            elif end == True and start == False:
                points[2] = [xStart, yStart]
        else:
            # Otherwise check to see if the point is similar to point[2]
            if abs(xStart - points[2][0]) < tolerance:
                points[2][0] = (points[2][0] + xStart) / 2
                points[2][1] = (points[2][1] + yStart) / 2
                counters[2] += 1
            elif abs(xEnd - points[2][0]) < tolerance:
                points[2][0] = (points[2][0] + xEnd) / 2
                points[2][1] = (points[2][1] + yEnd) / 2
                counters[2] += 1

    # Check to see which point has most votes, this should be the intersection point
    maxCount = max(counters)
    # points[2] must be the intersection point for the angle calculation to work
    if counters[0] == maxCount:
        temp = points[2]
        points[2] = points[0]
        points[0] = temp
    elif counters[1] == maxCount:
        temp = points[2]
        points[2] = points[1]
        points[1] = temp

    return points[0], points[1], points[2]

main()