import numpy as np

# Perform hough transform on an image
def houghTransform(image : np.ndarray, angleCount : int, edgeThreshold : float):
    height = image.shape[0]
    width = image.shape[1]
    
    # Length of the longest diagonal from corner to corner of the image
    diagonalLength = np.sqrt(width ** 2.0 + height ** 2.0)
    
    magnitudesCount = int(2 * diagonalLength)
    magnitudes = np.linspace(0.0, diagonalLength, magnitudesCount)
    
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
                    magnitude = y * np.cos(angle) + x * np.sin(angle)
                    magnitude = round(magnitude * (magnitudesCount / diagonalLength))
                    hough[int(magnitude), k] += 1
                    
                    # Pick a point to be used to check direction of line segment (picked 5th point arbitrarily)
                    if hough[int(magnitude), k] == 5:
                        voters[int(magnitude), k] = np.array([x, y])

                    
    return hough, magnitudes, angles, voters

# Get lineCount number of lines from the maxima sorted by number of votes
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
    while (abs(bestLines[0][1] - bestLines[1][1]) < 0.05):
        del maxima[0]
        bestLines = maxima[:2]
        
    return bestLines
