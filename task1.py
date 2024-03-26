import cv2
import numpy as np
import random

def main():
    for i in range(9, 21):
        image = cv2.imread("./angleData/image" + str(i) + ".png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cv2.imshow("image", image)
        cv2.waitKey(0)
        
        edges = findEdges(image)

        houghSpace, magnitudes, angles, voters = houghTransform(edges, 180, 0.0)
        lines = findMaxima(houghSpace, magnitudes, angles, voters)  

        angle, intersectionX, intersectionY = calculateAngle(lines)
        
        # Convert edges to color image
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Draw lines on the edges image
        for line in lines:
            r, theta, _, voter= line

            a = np.cos(theta)
            b = np.sin(theta)
            
            x0 = a*r
            y0 = b*r
            
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))

            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
 
            cv2.line(edges_color, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.line(edges_color, (int(intersectionX), int(intersectionY)), voter, (255, 0, 0), 2)
        
        # Draw intersection
        cv2.circle(edges_color, (int(intersectionX), int(intersectionY)), radius=3, color=(0, 255, 0), thickness=-1)
        
        print(angle)

        cv2.imshow("image", edges_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Calculate angle between lines in image
# Input: lines to calculate angle between
# Output: value of angle in degrees, intersection point of lines
def calculateAngle(lines : list):
        # Calculate angle between lines
        angle = abs(lines[0][1] - lines[1][1])
        
        # Find intersection of lines
        r1, theta1, _, voter1 = lines[0]
        r2, theta2, _, voter2 = lines[1]
        
        ax = r1 * np.cos(theta1)
        ay = r1 * np.sin(theta1)
        vx = np.sin(theta1)
        vy = -np.cos(theta1)
        
        bx = r2 * np.cos(theta2)
        by = r2 * np.sin(theta2)
        wx = np.sin(theta2)
        wy = -np.cos(theta2)
         
        t = (bx * wy - (by * wx) - (ax * wy) + (ay * wx)) / (vx * wy - (vy * wx))
        
        intersectionX = ax + t * vx
        intersectionY = ay + t * vy
        
        # Use points which voted for lines
        dot = np.dot(voter1 - (int(intersectionX), int(intersectionY)), voter2 - (int(intersectionX), int(intersectionY)))
        
        # Fix incorrect obtuse
        if ((dot < 0 and angle < np.pi / 2) or (dot > 0 and angle > np.pi / 2)):
            angle = np.pi - angle
            
        # Convert angle to degrees
        angle = angle * (180 / np.pi)
        
        return angle, intersectionX, intersectionY

# Use canny edge detection to identify edge in image
# Input: Image in the form of a numpy array
# Output: Edgemap in the form of a numpy array
def findEdges(image : np.ndarray, highFraction : float = 0.6, lowFraction : float = 0.2):
    # Find gradient
    sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Find magnitude of gradient
    intensity = np.sqrt(sobelX**2 + sobelY**2)
    intensity = intensity / np.max(intensity) * 255
    edges = intensity.astype(np.uint8)

    # Find direction of gradient
    direction = np.arctan2(sobelY, sobelX)

    # NON-MAX SUPPRESSION
    supressionMat = np.zeros_like(edges)

    for y in range(1, edges.shape[0] - 1):
        for x in range(1, edges.shape[1] - 1):
            
            f = 255
            b = 255

            angle = direction[y, x] * 180 / np.pi
            angle = abs(angle)

            if (angle >= 0 and angle < 22.5) or (angle >= 157.5 and angle <= 180):
                f = edges[y - 1, x + 1]
                b = edges[y + 1, x - 1]
            elif (angle >= 22.5 and angle < 67.5):
                f = edges[y - 1, x]
                b = edges[y + 1, x]
            elif (angle >= 67.5 and angle < 112.5):
                f = edges[y - 1, x - 1]
                b = edges[y + 1, x + 1]
            elif (angle >= 112.5 and angle < 157.5):
                f = edges[y, x - 1]
                b = edges[y, x + 1]
           
           
            if (edges[y, x] >= f) and (edges[y, x] >= b):
                supressionMat[y, x] = edges[y, x]

    edges = supressionMat

    # # DOUBLE THRESHOLD
    # highThreshold = edges.max() * highFraction
    # lowThreshold = edges.max() * lowFraction

    # thresholdMat = np.zeros_like(edges)

    # strongI = np.where(edges > highThreshold)
    # weakI = np.where((edges <= highThreshold) & (edges >= lowThreshold))

    # thresholdMat[strongI] = 255
    # thresholdMat[weakI] = 50

    # edges = thresholdMat

    
    # # HYSTERESIS

    # for y in range(1, edges.shape[0] - 1):
    #     for x in range(1, edges.shape[1] - 1):
    #         if (edges[y, x] == 50):
    #             if (edges[y - 1, x - 1] == 255) or (edges[y - 1, x] == 255) or \
    #             (edges[y - 1, x + 1] == 255) or (edges[y, x - 1] == 255) or \
    #             (edges[y, x + 1] == 255) or (edges[y + 1, x - 1] == 255) or \
    #             (edges[y + 1, x] == 255) or (edges[y + 1, x + 1] == 255):
    #                 edges[y, x] = 255
    #             else:
    #                 edges[y, x] = 0

    return edges


# Perform hough transform on an image
# Input: Edge image in the form of a numpy array, number of sample angles and threhold value for including edge
# Output: hough space array, with corresponding magnitudes and angles. And a random point which voted for any point in the hough space
def houghTransform(image : np.ndarray, angleCount : int, edgeThreshold : float):
    height = image.shape[0]
    width = image.shape[1]
    
    # Length of the longest diagonal from corner to corner of the image
    diagonalLength = np.sqrt(width ** 2.0 + height ** 2.0)
    
    magnitudesCount = int(4*diagonalLength)
    magnitudes = np.linspace(-diagonalLength, diagonalLength, magnitudesCount)
    
    angles = np.arange(0.0, np.pi, np.pi / angleCount)
     
    # Hough space array where votes will be accumulated
    hough = np.zeros((magnitudesCount, angleCount))
    voters = np.zeros((magnitudesCount, angleCount, 2), dtype=int)
    
    # Work on all the points in the image above a set threshold value
    for y in range(height):
        for x in range(width):
            if (image[y, x] > edgeThreshold):
                for k in range(len(angles)):
                    angle = angles[k]
                    magnitude = x * np.cos(angle) + y * np.sin(angle)
                    magnitude = round((magnitude + diagonalLength) * 2)
                    hough[int(magnitude), k] += 1
                    
                    # Pick a point to be used to check direction of line segment (picked 5th point arbitrarily)
                    if hough[int(magnitude), k] == 5:
                        voters[int(magnitude), k] = np.array([x, y])

                    
    return hough, magnitudes, angles, voters

# Get two lines from the maxima sorted by number of votes
# Input: hough space array, with corresponding magnitudes and angles. And a random point which voted for any point in the hough space
# Output: Two lines with the most votes
def findMaxima(hough : np.ndarray, magnitudes : np.ndarray, angles : np.ndarray, voters : np.ndarray):
    maxima = []
    for i in range(len(magnitudes)):
        for j in range(len(angles)):
            if (hough[i, j] > 0):
                maxima.append((magnitudes[i], angles[j], hough[i, j], voters[i, j]))
                    
    # Take top maxima
    maxima.sort(key = lambda x : x[2], reverse = True)
    
    # Discard lines which are duplicates
    bestLines = maxima[:2]
    while (abs(bestLines[0][1] - bestLines[1][1]) < 0.1):
        del maxima[0]
        bestLines = maxima[:2]
        
    return bestLines

def generateTestImage():
    # width = random.randrange(300, 1000)
    # height = random.randrange(300, 1000)
    width = 854
    height = 480
    
    intersection = np.array([random.randrange(20, width - 20), random.randrange(20, height - 20)])
    
    smallestAngle = 10
    angles = [0, 0]
    angles[0] = random.randrange(0, 360)
    angles[1] = random.randrange(angles[0] + smallestAngle, angles[0] + 180)
    
    angle = angles[1] - angles[0]
    
    # Create image
    #backgroundColour = (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255))
    backgroundColour = (0, 0, 0)
    image = np.full((height, width, 3), backgroundColour, np.uint8)
    
    # Ensure line colour different from background colour
    lineColour = (255, 255, 255)
    # lineColour = (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255))
    # while (lineColour == backgroundColour):
    #     lineColour = (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255))
    
    #lineThickness = random.randrange(2, 5)
    lineThickness = 1
    
    # Draw lines
    for i in range(2):
        length = random.randrange(30, 200)
        angles[i] *= np.pi / 180
        point = intersection + length * np.array([np.cos(angles[i]), np.sin(angles[i])])
        cv2.line(image, intersection, (int(point[0]), int(point[1])), lineColour, lineThickness)
        
    return image, angle

def createTestData(n, folder):
    textFile = open(folder +"/list.txt", "w")
    textFile.write("FileName,AngleInDegrees\n")
    
    for i in range(1, n + 1):
        image, angle = generateTestImage()
        cv2.imwrite(folder + "/image" + str(i) + ".png", image)
        textFile.write("image" + str(i) + ".png," + str(angle) + "\n")
        
    textFile.close()

if __name__=="__main__":
    #createTestData(20, "angleData")
    main()