from multiprocessing.reduction import duplicate
import numpy as np
from sympy import true

# Perform hough transform on an image
def houghTransform(image : np.ndarray, angleCount : int, edgeThreshold : float):
    width = image.shape[0]
    height = image.shape[1]
    
    # Length of the longest diagonal from corner to corner of the image
    diagonalLength = np.sqrt(width ** 2.0 + height ** 2.0)
    
    magnitudesCount = int(2 * diagonalLength)
    magnitudes = np.linspace(0.0, diagonalLength, magnitudesCount)
    
    angles = np.arange(0.0, np.pi, np.pi / angleCount)
     
    # Hough space array where votes will be accumulated
    hough = np.zeros((magnitudesCount, angleCount))
    
    # Work on all the points in the image above a set threshold value
    for j in range(height):
        for i in range(width):
            if (image[i, j] > edgeThreshold):
                # For each angle, calculate magnitude
                for k in range(len(angles)):
                    angle = angles[k]
                    magnitude = j * np.cos(angle) + i * np.sin(angle)
                    magnitude = round(magnitude * (magnitudesCount / diagonalLength))
                    # Add vote for line
                    hough[magnitude, k] += 1
                    
    return hough, magnitudes, angles

# Get lineCount number of lines from the maxima sorted by number of votes
def findMaxima(hough : np.ndarray, magnitudes : np.ndarray, angles : np.ndarray):
    maxima = []
    for i in range(len(magnitudes)):
        for j in range(len(angles)):
            if (hough[i, j] > 0):
                maxima.append((magnitudes[i], angles[j], hough[i, j]))
                    
    # Take top maxima
    maxima.sort(key = lambda x : x[2], reverse = True)
    
    # Discard lines which are duplicates
    bestLines = maxima[:2]
    while (abs(bestLines[0][1] - bestLines[1][1]) < 0.05):
        del maxima[0]
        bestLines = maxima[:2]
        
    return bestLines
