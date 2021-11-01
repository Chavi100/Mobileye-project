import os
import glob
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tfl_maneger.part1_api import find_tfl_lights


def find_a_not_tfl(gt_im):
    while True:
        not_tfl_y = random.randint(0, len(gt_im) - 5)
        not_tfl_x = random.randint(0, len(gt_im[0]) - 5)
        if gt_im[not_tfl_y, not_tfl_x] != 19:
            return (not_tfl_x, not_tfl_y)


def crop_image(image, x, y):
    left = max(x - 41, 0)
    right = min(x + 40, len(image[0]) - 1)
    top = max(y - 41, 0)
    bottom = min(y + 40, len(image) - 1)
    left_add = abs(min(x - 41, 0))
    right_add = max(x + 40 - (len(image[0]) - 1), 0)
    top_add = abs(min(y - 41, 0))
    bottom_add = max(y + 40 - (len(image) - 1), 0)
    image = image[top:bottom, left:right]
    result = np.hstack((np.hstack((np.zeros((len(image), left_add, 3)), image)), np.zeros((len(image), right_add, 3))))
    result = np.vstack(
        (np.vstack((np.zeros((top_add, len(result[0]), 3)), result)), np.zeros((bottom_add, len(result[0]), 3))))
    return result


def configure_image(image, path):
    print("========================")
    gt_im = image.replace('leftImg8bit', 'gtFine')[:-4] + '_labelIds.png'
    image = np.array(Image.open(image))
    gt_im = np.array(Image.open(gt_im))
    red_x, red_y, green_x, green_y = find_tfl_lights(image, some_threshold=42)
    cropped_im = []
    labels = []
    i = 0
    susp_to_tfl = []
    for x, y in zip(red_x + green_x, red_y + green_y):
        if gt_im[y, x] == 19:
            i += 1
            cropped_im.append(crop_image(image, x, y))
            labels.append(1)
        else:
            susp_to_tfl.append((x, y))
    susp = min(i // 2, len(susp_to_tfl))
    for j in range(susp):
        cropped_im.append(crop_image(image, susp_to_tfl[j][0], susp_to_tfl[j][1]))
        labels.append(0)
    for k in range(i - susp):
        not_tfl = find_a_not_tfl(gt_im)
        cropped_im.append(crop_image(image, not_tfl[0], not_tfl[1]))
        labels.append(0)

    return cropped_im, labels


def bin_file(data, labels, path):
    data_array = np.array(data, dtype='uint8')
    f = open(f'db/Net dataset/{path}/data.bin', "ab")
    data_array.tofile(f)
    f.close()
    label_array = np.array(labels, dtype='uint8')
    f = open(f'db/Net dataset/{path}/labels.bin', "ab")
    label_array.tofile(f)
    f.close()


def prepare_set(path):
    set = f'./db/leftImg8bit/{path}/'

    set_images = glob.glob(os.path.join(set, '*/*_leftImg8bit.png'))
    cropped_im = []
    labels = []
    for im in set_images:
        image_cropped, image_labels = configure_image(im, path)
        cropped_im += image_cropped
        labels += image_labels
    bin_file(cropped_im, labels, path)


def show_label(path, index):
    data_file = np.memmap(f'db/Net dataset/{path}/data.bin', offset=index * (81 * 81 * 3), shape=(81, 81, 3))
    label_file = np.memmap(f'db/Net dataset/{path}/labels.bin', offset=index, shape=(1,))
    plt.imshow(data_file)
    if label_file == 0:
        plt.title("no traffic light")
    else:
        plt.title("traffic light")
    plt.show()
    print(label_file)


if __name__ == '__main__':
    prepare_set('train')
    prepare_set('val')
    pass