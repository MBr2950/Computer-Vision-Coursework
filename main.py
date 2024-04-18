import argparse
import task3
import pandas as pd
import cv2

import task1, task2, task3

def testTask1(folderName):
    print("Task 1 Start")

    # assume that this folder name has a file list.txt that contains the annotation
    task1Data = pd.read_csv(folderName + "/list.txt")
    images = task1Data["FileName"].tolist()
    testAngles = task1Data["AngleInDegrees"].tolist()
    
    predAngles = []
    totalError = 0
    
    for i in range(0, len(images)):
        image = cv2.imread(folderName + "/" + images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        edges = task1.findEdges(image)
        houghSpace, magnitudes, angles, voters = task1.houghTransform(edges, 720, 0.0)
        lines = task1.findMaxima(houghSpace, magnitudes, angles, voters)  

        angle, _, _, _ = task1.calculateAngle(lines)
        predAngles.append(angle)
        error = abs(angle - testAngles[i])
        totalError += error
        
        #print(images[i] + ", Predicted angle:" + str(round(angle)), "Actual angle:" + str(testAngles[i]), "Error:" + str(round(error)))
        print(f"Image {i} Error: { str(round(error))}")
    
    print("Total Error:", totalError, "\n")
    return (totalError)

def testTask2(iconDir, testDir):
    print("Task 2 Start")

    # assume that test folder name has a directory annotations with a list of csv files
    # load train images from iconDir and for each image from testDir, match it with each class from the iconDir to find the best match
    # For each predicted class, check accuracy with the annotations
    # Check and calculate the Intersection Over Union (IoU) score
    # based on the IoU determine accuracy, TruePositives, FalsePositives, FalseNegatives
    
    # Assuming that the icons and images folders are layed out in the same way as the example data
    # and argument is of format ./IconDataset
    iconsLocation = iconDir + "/png/"
    imagesLocation = testDir + "/images/"
    annotationsLocation = testDir + "/annotations/"

    truePositives, falsePositives, falseNegatives, _, _ = task2.main(imagesLocation, iconsLocation, annotationsLocation)

    accuracy = truePositives / (truePositives + falsePositives + falseNegatives)
    accuracy *= 100 
    
    return (accuracy, truePositives, falsePositives, falseNegatives)


def testTask3(iconFolderName, testFolderName):
    print("Task 3 Start")

    # assume that test folder name has a directory annotations with a list of csv files
    # load train images from iconDir and for each image from testDir, match it with each class from the iconDir to find the best match
    # For each predicted class, check accuracy with the annotations
    # Check and calculate the Intersection Over Union (IoU) score
    # based on the IoU determine accuracy, TruePositives, FalsePositives, FalseNegatives

    # Assuming that the icons and images folders are layed out in the same way as the example data
    # and argument is of format './IconDataset'
    iconsLocation = iconFolderName + "/png/"
    imagesLocation = testFolderName + "/images/"
    annotationsLocation = testFolderName + "/annotations/"
    truePositives, falsePositives, falseNegatives, _, _ = task3.main(iconsLocation=iconsLocation, imagesLocation=imagesLocation, annotationsLocation=annotationsLocation)
    
    accuracy = truePositives / (truePositives + falsePositives + falseNegatives)
    accuracy *= 100 

    return (accuracy, truePositives, falsePositives, falseNegatives)


if __name__ == "__main__":
    # parsing the command line path to directories and invoking the test scripts for each task
    parser = argparse.ArgumentParser("Data Parser")
    parser.add_argument("--Task1Dataset", help="Provide a folder that contains the Task 1 Dataset.", type=str, required=False)
    parser.add_argument("--IconDataset", help="Provide a folder that contains the Icon Dataset for Task2 and Task3.", type=str, required=False)
    parser.add_argument("--Task2Dataset", help="Provide a folder that contains the Task 2 test Dataset.", type=str, required=False)
    parser.add_argument("--Task3Dataset", help="Provide a folder that contains the Task 3 test Dataset.", type=str, required=False)
    args = parser.parse_args()
    if(args.Task1Dataset!=None):
        # This dataset has a list of png files and a txt file that has annotations of filenames and angle
        testTask1(args.Task1Dataset)
    if(args.IconDataset!=None and args.Task2Dataset!=None):
        # The Icon dataset has a directory that contains the icon image for each file
        # The Task2 dataset directory has two directories, an annotation directory that contains the annotation and a png directory with list of images 
        testTask2(args.IconDataset,args.Task2Dataset)
    if(args.IconDataset!=None and args.Task3Dataset!=None):
        # The Icon dataset directory contains an icon image for each file
        # The Task3 dataset has two directories, an annotation directory that contains the annotation and a png directory with list of images 
        testTask3(args.IconDataset,args.Task3Dataset)
