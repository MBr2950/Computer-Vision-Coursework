from task2funcs import *
import math

SCALE = 1 / math.sqrt(2)
SPEED_SCALES = 5
ICON_ROTATIONS = 1
ICON_START_SCALE = 2
ICON_SCALES = 5
THRESHOLD = 30

    
def main(testDir, iconDir, annotationsDir, outputDir):
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
    print(f"True Positives: {truePositives}, False Positives: {falsePositives}, False Negatives: {falseNegatives}, Average IoU: {averageIntersectionOverUnion}")


if __name__ == '__main__':
    iconDir = "IconDataset/png/"
    testDir = "Task2Dataset/images/"
    annotationsDir = "Task2Dataset/annotations/"
    outputDir = "Task2OutputImages/"
    main(testDir, iconDir, annotationsDir, outputDir)