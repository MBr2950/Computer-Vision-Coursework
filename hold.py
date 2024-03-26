def build_gaussian_pyramid(image, num_scales):
    pyramid = [image]
    for i in range(num_scales):
        # Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(pyramid[i], (5, 5), 0)
        # Subsample the blurred image
        subsampled_image = cv2.pyrDown(blurred_image)
        pyramid.append(subsampled_image)
    return pyramid