from task2funcs import *
import math

SCALE = 1 / math.sqrt(2)
SPEED_SCALES = 5
ICON_ROTATIONS = 1
ICON_START_SCALE = 2
ICON_SCALES = 5
THRESHOLD = 30

    
def main():
    print("Loading icons...")
    iconDir = "IconDataset/png/"
    icons = load_and_preprocess_images(iconDir, ICON_ROTATIONS)
    print("Icons loaded.")

    print("Loading test images...")
    testDir = "Task2Dataset/images/"
    testImages = load_and_preprocess_images(testDir)
    print("Test images loaded.")

    print("Predicting...")
    results = predict(testImages, icons, SCALE, SPEED_SCALES, ICON_START_SCALE, ICON_SCALES, THRESHOLD)
    print("Prediction complete.")

    print(results)

    plot_results(testImages, results)


if __name__ == '__main__':
    main()