import cv2
import numpy as np
import random


def save_image(path: str, image):
    cv2.imwrite(path, image)


def read_image(path: str):
    result_image = cv2.imread(path, 3)

    return result_image


def grayscale(image):
    result_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return result_image


def blur(image, blur_factor):
    blur_factor -= 1 if blur_factor % 2 == 0 else 0
    result_image = cv2.GaussianBlur(image, (blur_factor, blur_factor), 0)

    return result_image


def off_center(image, x: int, y: int):
    dimensions = image.shape[0]
    m = np.float32([[1, 0, x], [0, 1, y]])
    result_image = cv2.warpAffine(image, m, (dimensions, dimensions), cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return result_image


def rotate(image, angle: int):
    dimensions = image.shape[0]

    m = cv2.getRotationMatrix2D((dimensions / 2, dimensions / 2), angle, 1)
    result_image = cv2.warpAffine(image, m, (dimensions, dimensions), cv2.BORDER_CONSTANT)

    return result_image


def zoom_in(image, new_dimensions: int):
    # Setup parameters and crop out part of the image.
    original_dimensions = image.shape[0]
    result_image = set_dimensions(crop_image(image, new_dimensions), original_dimensions)

    return result_image


def zoom_out(image, new_dimensions: int):
    """Method zooms out of image.

    :param new_dimensions: Size to which the original is reduced to. (Must be > 0)
    """

    original_dimensions = image.shape[0]

    # Setup blank image where downsized image will be inserted.
    result_image = np.zeros((original_dimensions, original_dimensions, 3), np.uint8)
    result_image[:, :] = (0, 0, 0)

    if new_dimensions <= 0:
        print("New size has to be bigger than zero.")
        return image

    # Downsize image to 'new_size' parameter.
    shrunken_image = set_dimensions(image, new_dimensions)

    # Setup parameters for insertion and perform it.
    x_0, x_1 = (int(original_dimensions / 2 - new_dimensions / 2), int(original_dimensions / 2 + new_dimensions / 2))
    result_image[x_0: x_1, x_0: x_1] = shrunken_image

    return result_image


def set_dimensions(image, new_dimensions: int):
    """Method changes dimensions of given image.

    :param new_dimensions: New dimensions of the image.
    :param image: Image
    """
    result_image = cv2.resize(image, (new_dimensions, new_dimensions))

    return result_image


def crop_image(image, new_dimensions: int):
    original_dimensions = image.shape[0]
    start = int((original_dimensions - new_dimensions) / 2)
    end = original_dimensions - start
    cropped_image = image[start: end, start: end]

    return cropped_image


def getSobel(channel):
    """Performs Sobel operator on channel

    Performs Sobel operator on channel, which creates an image emphasising edges.
    This improves the results of the edge detection algorithm.

    :param channel: Channel, that Sobel operation is performed on.
    :return: Processed image.
    """

    sobel_x = cv2.Sobel(channel, cv2.CV_16S, 1, 0, borderType=cv2.BORDER_REPLICATE)
    sobel_y = cv2.Sobel(channel, cv2.CV_16S, 0, 1, borderType=cv2.BORDER_REPLICATE)
    sobel = np.hypot(sobel_x, sobel_y)

    return sobel


def find_significant_contours(sobel_8u):
    """Finds significant contours in the image to determine the edges of an object.

    :param sobel_8u: Image that algorithm is performed on.
    :return: Significant contours.
    """

    image, contours, hierarchy = cv2.findContours(sobel_8u, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find level 1 contours
    level1 = []
    for i, tuple in enumerate(hierarchy[0]):
        # Each array is in format (Next, Prev, First child, Parent)
        # Filter the ones without parent
        if tuple[3] == -1:
            tuple = np.insert(tuple, 0, [i])
            level1.append(tuple)

    # From among them, find the contours with large surface area.
    significant = []
    # If contour isn't covering 0.5% of total area of image then it probably is too small.
    too_small = sobel_8u.size * 5 / 1000
    for tuple in level1:
        contour = contours[tuple[0]]
        area = cv2.contourArea(contour)
        if area > too_small:
            significant.append([contour, area])

    significant.sort(key=lambda x: x[1])

    return [x[0] for x in significant]


def diff(pixel_0: (int, int, int), pixel_1: (int, int, int)):
    """Calculates the difference in color between two pixels.

    :param pixel_0: Pixel 0.
    :param pixel_1: Pixel 1.
    :return: Absolute difference between the pixels.
    """

    return abs(pixel_0[0] - pixel_1[0]) + abs(pixel_0[1] - pixel_1[1]) + abs(pixel_0[2] - pixel_1[2])


def detect_object(img, bg_color: (int, int, int)):
    """Detects piece in image and returns its coordinates
    Method checks image for piece. Returns coordinates which can draw a the smallest possible rectangle
    containing the piece.

    :param img: Image, which contains the object.
    :param bg_color: Background color of the image.
    :return: Four-tuple with the coordinates of the piece.
    """

    img_size = img.shape[0]
    # Initialize coordinate variables.
    x_0, y_0, x_1, y_1 = 0, 0, img_size, img_size
    # Variables indicate whether coordinates where found.
    x_0_found, y_0_found, x_1_found, y_1_found = False, False, False, False
    for i in range(0, img_size - 6, 5):
        for j in range(0, img_size - 6, 5):
            if not x_0_found and diff(img[j][i], bg_color) > 300:
                x_0 = i
                x_0_found = True
            if not y_0_found and diff(img[i][j], bg_color) > 300:
                y_0 = i
                y_0_found = True
            if not x_1_found and diff(img[img_size - j - 1][img_size - i - 1], bg_color) > 300:
                x_1 = img_size - i - 1
                x_1_found = True
            if not y_1_found and diff(img[img_size - i - 1][img_size - j - 1], bg_color) > 300:
                y_1 = img_size - i - 1
                y_1_found = True
            if x_0_found and y_0_found and x_1_found and y_1_found:

                x_0 -= 30 if x_0 - 30 > 0 else 0
                y_0 -= 30 if y_0 - 30 > 0 else 0
                x_1 += 30 if x_1 + 30 < img_size else 0
                y_1 += 30 if x_1 + 30 < img_size else 0

                return x_0, y_0, x_1, y_1

    # Default case, when no object could be detected.
    return 0, 0, img_size, img_size


def calculate_avg_bg_color_with_mask(img, mask):
    """Method calculates average color around the detected object.
    Calculates average color of all color values at the edges of the true area in the mask.

    :param img: Image, that will be used.
    :param mask: Mask describing the object.
    :return: Average background color around the object.
    """

    img_size = img.shape[0]
    r, g, b, number_samples = 0, 0, 0, 0
    # Add up color values of all samples and remember number of samples.
    for i in range(0, img_size - 1, 5):
        for j in range(0, img_size - 1, 5):
            if mask[i][j] and (not mask[i - 1][j] or not mask[i + 1][j] or not mask[i][j - 1] or not mask[i][j + 1]):
                sample_b, sample_g, sample_r = img[i][j]
                r, g, b = r + sample_r, g + sample_g, b + sample_b
                number_samples += 1

    # Calculate average and returns it.
    return int(b / number_samples), int(g / number_samples), int(r / number_samples)


def fill_horizontal_img(img_large, img_small):
    """Method fills a larger image up with multiple copies of a smaller image of the same width.

    The two images need to be of the same width. The smaller image is inserted multiple times
    in the larger image in order to fill the larger image up.

    :param img_large: Larger image to be filled up.
    :param img_small: Smaller image used to fill up large image.
    :return: Large image filled up with copies of small image.
    """

    img_large_size = img_large.shape[0]
    img_small_size = img_small.shape[0]

    filled = 0
    # Repeat inserting small image until the filled size is larger then the large image size.
    while filled <= img_large_size:
        # Mirror small image horizontally.
        img_small = cv2.flip(img_small, 0)

        # Calculate values, describing what of the small image is inserted, and where it is
        # inserted in the large image.
        from_y_0 = max(img_large_size - img_small_size - filled, 0)
        to_y_0 = img_large_size - filled
        from_y_1 = max((img_small_size - (img_large_size - filled)), 0)
        to_y_1 = img_small_size

        # Insert small image in large image.
        img_large[from_y_0: to_y_0, :] = img_small[from_y_1: to_y_1, :]

        # Update size of filled area.
        filled += img_small_size

    return img_large


def fill_vertical_img(img_large, img_small):
    """Method fills a larger image up with multiple copies of a smaller image of the same height.

    The two images need to be of the same height. The smaller image is inserted multiple times
    in the larger image in order to fill the larger image up.

    :param img_large: Larger image to be filled up.
    :param img_small: Smaller image used to fill up large image.
    :return: Large image filled up with copies of small image.
    """

    img_large_size = img_large.shape[1]
    img_small_size = img_small.shape[1]

    filled = 0
    # Repeat inserting small image until the filled size is larger then the large image size.
    while filled <= img_large_size:
        # Mirror small image vertically.
        img_small = cv2.flip(img_small, 1)

        # Calculate values, describing what of the small image is inserted, and where it is
        # inserted in the large image.
        from_x_0 = max(img_large_size - img_small_size - filled, 0)
        to_x_0 = img_large_size - filled
        from_x_1 = max((img_small_size - (img_large_size - filled)), 0)
        to_x_1 = img_small_size

        # Insert small image in large image.
        img_large[:, from_x_0: to_x_0] = img_small[:, from_x_1: to_x_1]

        # Update size of filled area.
        filled += img_small_size

    return img_large


def enlarge_background(img, new_img_size: int, x_0: int, y_0: int, x_1: int, y_1: int):
    """Method embeds object in larger image with extended background."""

    # Ensure borders at all size of the enlarged images have same size.
    while True:
        img_size = img.shape[0]
        b_size = int((new_img_size - img_size) / 2)

        if (new_img_size - img_size) % 2 == 0:
            break
        else:
            img = img[0: (img_size - 1), 0: (img_size - 1)]

    # Create new empty enlarged image and insert original image in the middle.
    new_img = np.zeros((new_img_size, new_img_size, 3), np.uint8)
    new_img[b_size: (b_size + img_size), b_size: b_size + img_size] = img

    # Setup left border of the new image with the left border of the original image.
    border_left = np.zeros((img_size, b_size, 3), np.uint8)
    left_cropped = img[0: img_size, 0: x_0]
    border_left = fill_vertical_img(img_large=border_left, img_small=left_cropped)

    # Setup right border of the new image with the right border of the original image.
    border_right = np.zeros((img_size, b_size, 3), np.uint8)
    right_cropped = cv2.flip(img[0: img_size, x_1: img_size], 1)
    border_right = cv2.flip(fill_vertical_img(img_large=border_right, img_small=right_cropped), 1)

    # Insert the created right and left border to the new image.
    new_img[b_size: (b_size + img_size), 0: b_size] = border_left
    new_img[b_size: (b_size + img_size), b_size + img_size: new_img_size] = border_right

    # Setup top border of the new image with the top border of the original image.
    border_top = np.zeros((b_size, new_img_size, 3), np.uint8)
    top_cropped = new_img[b_size: (b_size + y_0), 0: new_img_size]
    border_top = fill_horizontal_img(img_large=border_top, img_small=top_cropped)

    # Setup bottom border of the new image with the bottom border of the original image.
    border_bottom = np.zeros((b_size, new_img_size, 3), np.uint8)
    bottom_cropped = cv2.flip(new_img[(y_1 + b_size): (img_size + b_size), 0: new_img_size], 0)
    border_bottom = cv2.flip(fill_horizontal_img(img_large=border_bottom, img_small=bottom_cropped), 0)

    # Insert the created top and bottom border to the new image.
    new_img[0: b_size, 0: new_img_size] = border_top
    new_img[(b_size + img_size): new_img_size, 0: new_img_size] = border_bottom

    return new_img


def calculate_avg_bg_color(img):
    """Calculates average background color of the image.

    Method takes four samples from each corner of the image and returns the average
    color of the samples.

    :param img: Image of which the average background color is calculated of.
    :return: Average Background color.
    """

    img_size = img.shape[0]

    # Take four sample pixels from each corner of the image
    s_0, s_1 = img[1, 1], img[1, (img_size - 1)]
    s_2, s_3 = img[(img_size - 1), 1], img[(img_size - 1), (img_size - 1)]

    # Calculate their average rgb value.
    avg_b = int((int(s_0[0]) + int(s_1[0]) + int(s_2[0]) + int(s_3[0])) / 4)
    avg_g = int((int(s_0[1]) + int(s_1[1]) + int(s_2[1]) + int(s_3[1])) / 4)
    avg_r = int((int(s_0[2]) + int(s_1[2]) + int(s_2[2]) + int(s_3[2])) / 4)

    return avg_b, avg_g, avg_r


def adjust_brightness(image, lighting_factor):
    dimensions = image.shape[0]
    for x in range(0, dimensions):
        for y in range(0, dimensions):
            image[x, y][0] = max(min(image[x, y][0] + lighting_factor + get_noise(5), 255), 0)
            image[x, y][1] = max(min(image[x, y][1] + lighting_factor + get_noise(5), 255), 0)
            image[x, y][2] = max(min(image[x, y][2] + lighting_factor + get_noise(5), 255), 0)

    return image


def get_noise(boundary: int):
    return int(random.random() * boundary + 1)


def adjust_contrast_exposure(img, c: int, e: int):
    """Method adjust contrast and exposure of image.

    :param e: Determines the change in exposure.
    :param c: Determines the change in contrast.
    :param img: Original image.
    :return: Image with adjusted contrast and exposure.
    """

    adjusted_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    adjusted_img[:, :, 2] = [[max(pixel - c, 0) if pixel < 190 else min(pixel + c + e, 255) for pixel in row] for row in
                             adjusted_img[:, :, 2]]

    return cv2.cvtColor(adjusted_img, cv2.COLOR_HSV2BGR)
