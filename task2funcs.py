import cv2
import os
from scipy import ndimage
import numpy as np
import math
import csv


# loads list of images given relative dir filepath
def load_images(dir):
    files = [dir + file for file in os.listdir(dir) if file.endswith('.png')]
    images = []
    for file in files:
        image = cv2.imread(file)
        images.append((file.split('/')[-1], image))
    return images


# loads list of annotations given relative dir filepath
def load_annotations(dir):
        annotations = []
        annotationPaths = os.listdir(dir)
        # Total number of icons, used for calculating accuracy
        numIcons = 0

        for i in range(len(annotationPaths)):
            annotation = []

            file = open(dir + annotationPaths[i], mode ='r')
            lines = csv.reader(file)
            for line in lines:
                annotation.append(line)
            annotation = annotation[1:]
            numIcons += len(annotation)
            file.close()

            annotations.append(annotation)
        return annotations, numIcons


def preprocess_image(image):
    # Convert all test images to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert background to 0
    image = cv2.threshold(image, 255, 0, cv2.THRESH_TOZERO_INV)[1]
    return image


def load_and_preprocess_images(dir, rotations = 1):
    images = load_images(dir)
    images = [(name, preprocess_image(image)) for name, image in images]

    for i in range(1, rotations):
        angle = i * 360 / rotations
        images += [(name + "_rot" + str(angle), ndimage.rotate(image, angle)) for name, image in images]

    return images




def build_gaussian_pyramid(image, num_scales, scale_factor, start_scale = 0):
    for _ in range(start_scale):
        image = cv2.GaussianBlur(image, (3, 3), 0)
        image = ndimage.zoom(image, scale_factor)
    
    pyramid = [image]
    for _ in range(num_scales - 1):
        image = cv2.GaussianBlur(image, (3, 3), 0)
        #image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
        image = ndimage.zoom(image, scale_factor)
        pyramid.insert(0, image)
    return pyramid





def hone(image_pyramid, template_pyramid, scale_factor, best_loc):

    assert(len(image_pyramid) == len(template_pyramid))

    min_rss = np.inf
    best_size = (0, 0)

    for image_level, template_level in zip(image_pyramid, template_pyramid):

        t_size = template_level.shape
        t_norm = 1 / (t_size[0] * t_size[1])

        width = image_level.shape[1] - t_size[1] + 1
        height = image_level.shape[0] - t_size[0] + 1

        startX = max(math.floor((best_loc[0] - 1) / scale_factor), 0)
        endX = min(math.ceil((best_loc[0] + 1) / scale_factor), width - 1)
        startY = max(math.floor((best_loc[1] - 1) / scale_factor), 0)
        endY = min(math.ceil((best_loc[1] + 1) / scale_factor), height - 1)


        min_rss = np.inf
        best_size = (0, 0)
        best_loc = None


        for y in range(startY, endY + 1):
            for x in range(startX, endX + 1):
                section = image_level[y : y + t_size[0], x : x + t_size[1]]
                rss = np.sum((section - template_level) ** 2) * t_norm
                if rss < min_rss:
                    min_rss = rss
                    best_size = t_size
                    best_loc = (x, y)

    return min_rss, best_loc, best_size



# returns the location of the best match and the map of the RSS values
def match_template_rss(image_pyramid, template_pyramid, scale_factor):
        
    start_level = image_pyramid[0]
    image_pyramid = image_pyramid[1:]


    min_rss = np.inf
    best_size = (0, 0)
    best_loc = None
    best_template_i = 0

    for i, template_level in enumerate(template_pyramid[:-len(image_pyramid)]):

        t_size = template_level.shape
        t_norm = 1 / (t_size[0] * t_size[1])

        width = start_level.shape[1] - t_size[1] + 1
        height = start_level.shape[0] - t_size[0] + 1


        for y in range(height):
            for x in range(width):
                section = start_level[y : y + t_size[0], x : x + t_size[1]]
                rss = np.sum((section - template_level) ** 2) * t_norm
                if rss < min_rss:
                    min_rss = rss
                    best_size = t_size
                    best_loc = (x, y)
                    best_template_i = i

    if len(image_pyramid) > 0:
        i = best_template_i
        min_rss, best_loc, best_size = hone(image_pyramid, template_pyramid[i + 1:i + 1 + len(image_pyramid)], scale_factor, best_loc)

    
    return min_rss, best_loc, best_size



def predict(images, icons, annotations, numIcons, SCALE, SPEED_SCALES, ICON_START_SCALE, ICON_SCALES, THRESHOLD):
    print("Building pyramids...")
    icon_pyramids = [(name, build_gaussian_pyramid(icon, ICON_SCALES + SPEED_SCALES - 1, SCALE, ICON_START_SCALE)) for name, icon in icons]
    test_pyramids = [(name, build_gaussian_pyramid(img, SPEED_SCALES, SCALE)) for name, img in images]
    print("Pyramids built.")

    results = {}

    for i in range(len(test_pyramids)):
        test_name, test_pyramid = test_pyramids[i]

        # Finds icons in each image
        print("Processing image", test_name)
        matches = []
        for icon_name, icon_pyramid in icon_pyramids:
            #print(icon_name)
            score, top_left, dim = match_template_rss(test_pyramid, icon_pyramid, SCALE)
            if score <= THRESHOLD:
                matches.append([icon_name, score, top_left, dim])

        results[test_name] = matches

    return results


# Calculates the 'IntersectionOverUnion' metric, a measure of what percentage of the bounding box
# identified lines up with the actual bounding box
# Inputs: First box, in the format (x1, y1, x2, y2); second box, in the format (x1, y1, x2, y2) 
def calculateIntersectionOverUnion(box1, box2):
    # 0 are top left coordinates, 1 are bottom right coordinates
    XA0, YA0, XA1, YA1 = box1
    XB0, YB0, XB1, YB1 = box2

    # Finds the coordinates of the overlap between the boxes
    X0Inter = max(XA0, XB0)
    Y0Inter = max(YA0, YB0)
    X1Inter = max(XA1, XB1)
    Y1Inter = max(YA1, YB1)

    # If the width of the height are negative, the boxes don't overlap at all
    intersectionWidth = (X1Inter - X0Inter)
    intersectionHeight = (Y1Inter - Y0Inter)
    
    # Calculate overlap (intersection over union)
    if intersectionWidth >= 0 and intersectionHeight >= 0: 
        intersectionArea = intersectionWidth * intersectionHeight
        unionArea = (XA1 - XA0) * (YA1 - YA0) + (XB1 - XB0) * (YB1 - YB0) - intersectionArea

        intersectionOverUnion = intersectionArea / unionArea
    else:
        intersectionOverUnion = 0

    return intersectionOverUnion


def plot_results(images, annotations, numIcons, results, outputDir):
    # Used for measuring accuracy
    truePositives = 0
    falsePositives = 0
    falseNegatives = 0
    totalIntersectionOverUnion = 0

    images = [images for images in images if images[0] in results]
    for i in range(len(images)):
        name, image = images[i]
        annotation = annotations[i]

        # print("\nDisplaying", name)

        image_results = results[name]
        for j in range(len(image_results)):
            match = image_results[j]
            print("Name:", match)
            icon_name, score, top_left, dim = match
            bottom_right = (top_left[0] + dim[0], top_left[1] + dim[1])

            tag = icon_name + "," + str(score)
            
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), 2)
            cv2.putText(image, tag, (top_left[0], max(top_left[1] - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Checks if the match is correct, to add to true positives or false positives
            truePositive = False
            for line in annotation:
                # Check if predicted icon name matches name in annotation, accounting for changes in naming
                if icon_name[1:-4] == line[0]:
                    x1 = top_left[0]
                    y1 = top_left[1]
                    x2 = bottom_right[0]
                    y2 = bottom_right[1]
                    intersectionOverUnion = calculateIntersectionOverUnion((x1, y1, x2, y2), (int(line[1]), int(line[2]), int(line[3]), int(line[4])))
                    totalIntersectionOverUnion += intersectionOverUnion

                    if intersectionOverUnion > 0.5:
                        truePositives += 1
                        truePositive = True

            if truePositive == False:
                falsePositives += 1

        # Finds number of icons not detected, and average intersectionOverUnion (including false negatives,
        # on average how much do predicted bounding boxes line up with actual bounding boxes)
        falseNegatives = numIcons - truePositives
        averageIntersectionOverUnion = (totalIntersectionOverUnion / (truePositives + falsePositives + falseNegatives)) * 100

        # cv2.imshow(name, image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Saves the image, if is of right type
        if type(image) == np.ndarray:
            cv2.imwrite(outputDir + name, image)
            # print("Image", i + 1, "done")

    return truePositives, falsePositives, falseNegatives, averageIntersectionOverUnion