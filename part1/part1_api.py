try:
    import os
    import json
    import glob
    import argparse

    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage.filters import maximum_filter

    from PIL import Image

    import matplotlib.pyplot as plt

    from skimage import img_as_float
    from skimage.feature import peak_local_max
except ImportError:
    print("Need to fix the installation")
    raise

def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    c_image = img_as_float(c_image)
    red_image = c_image[:, :, 0]
    green_image = c_image[:, :, 1]

    # high_pass_kernel = [[-1 / 9, -1 / 9, -1 / 9], [-1 / 9, 8 / 9, -1 / 9], [-1 / 9, -1 / 9, -1 / 9]]
    # red_image = sg.convolve(red_image, high_pass_kernel, mode='same')
    # green_image = sg.convolve(green_image, high_pass_kernel, mode='same')

    kernel = np.array([[1/2, 1.0, 1 / 4, 1 / 4, 1 / 8, 1 / 8],
                       [1.0, 1 / 4, 1 / 4, 1 / 8, 1 / 8, -1 / 2],
                       [1 / 4, 1 / 4, 1 / 8, 1 / 8, 1 / 8, -1 / 2],
                       [1 / 8, 1 / 8, 1 / 8, -1 / 2, -1 / 2, -1.0],
                       [1 / 8, 1 / 8, -1 / 2, -1 / 2, -1.0, -1/2]])

    red_image = sg.convolve(red_image, kernel, mode='same')
    green_image = sg.convolve(green_image, kernel, mode='same')
    red_coordinates = peak_local_max(red_image, min_distance=50, num_peaks=10)
    green_coordinates = peak_local_max(green_image, min_distance=50, num_peaks=10)

    red_x = []
    red_y = []
    green_x = []
    green_y = []
    for crd in red_coordinates:
        pixel_rgb = c_image[crd[0]][crd[1]]
        if pixel_rgb[0] >= pixel_rgb[1] and pixel_rgb[0] > pixel_rgb[2]:
            red_x.append(crd[1])
            red_y.append(crd[0])
    for crd in green_coordinates:
        pixel_rgb = c_image[crd[0]][crd[1]]
        if pixel_rgb[1] >= pixel_rgb[0]:
            green_x.append(crd[1])
            green_y.append(crd[0])

    return red_x, red_y, green_x, green_y


### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]
    show_image_and_gt(image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(image, some_threshold=42)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    # default_base = './db/leftImg8bit/test/bonn'
    default_base = 'images'

    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))

    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')

        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)

    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()
