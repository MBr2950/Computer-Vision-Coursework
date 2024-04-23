from task2funcs import *
import math
import time

SCALE = 1 / math.sqrt(2)
SPEED_SCALES = 6
ICON_ROTATIONS = 1
ICON_START_SCALE = 2
ICON_SCALES = 5
THRESHOLD = 30

    
def main(testDir, iconDir, annotationsDir, outputDir = "Task2OutputImages/"):
    startTime = time.time()

    # Create copy of images, for final display
    images = load_images(testDir)
    images = [(name, image) for name, image in images]

    # Get information about images
    annotations, numIcons = load_annotations(annotationsDir)

    print("Loading icons...")
    icons = load_and_preprocess_images(iconDir, ICON_ROTATIONS)
    print("Icons loaded.")

    print("Loading test images...")
    testImages = load_and_preprocess_images(testDir)
    print("Test images loaded.")

    print("Predicting...")
    results = predict(testImages, icons, annotations, numIcons, SCALE, SPEED_SCALES, ICON_START_SCALE, ICON_SCALES, THRESHOLD)
    print("Prediction complete.")

    truePositives, falsePositives, falseNegatives, averageIntersectionOverUnion = plot_results(images, annotations, numIcons, results, outputDir)
    runtime = time.time() - startTime
    print(f"True Positives: {truePositives}, False Positives: {falsePositives}, False Negatives: {falseNegatives}, Average IoU: {averageIntersectionOverUnion}, Runtime: {runtime} \n")

    return truePositives, falsePositives, falseNegatives, averageIntersectionOverUnion, runtime




if __name__ == '__main__':
    iconDir = "IconDataset/png/"
    testDir = "Task2Dataset/images/"
    annotationsDir = "Task2Dataset/annotations/"
    truePositives, falsePositives, falseNegatives, averageIntersectionOverUnion, runtime = main(testDir, iconDir, annotationsDir)

    accuracy = truePositives / (truePositives + falsePositives + falseNegatives)
    accuracy *= 100

    truePositiveRate = truePositives / (truePositives + falseNegatives)
    truePositiveRate *= 100

    print(f"Accuracy: {accuracy}, True Positive Rate: {truePositiveRate}")