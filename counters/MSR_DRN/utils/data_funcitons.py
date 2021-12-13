import numpy as np
import torch

def create_gausian_mask(center_point, nCols, nRows, radius = (5,5), q = 99):
    '''
    create_gausian_mask creates a gaussian mask to be used as GT annotations for the detection-based counter
    :param center_point:
    :param nCols:
    :param nRows:
    :param q:
    :param s:
    :param radius:
    :return:
    '''
    s = 3
    if (s>=radius[0]):
        s=1
    x = np.tile(range(nCols), (nRows,1))
    y = np.tile(np.reshape(range(nRows),(nRows,1)),(1,nCols))

    x2 = (((x - round(center_point[0].item()))*s) / radius[0]) ** 2
    y2 = (((y - round(center_point[1].item()))*s) / radius[1]) ** 2

    p = np.exp(-0.5 * (x2 + y2))

    p[np.where(p < np.percentile(p, q))] = 0

    p = p / np.max(p)
    if not np.isfinite(p).all() or not np.isfinite(p).all():
        print('divide by zero')
    return torch.from_numpy(p)



def read_annotations_ON(csv_reader):
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            img_file, num_of_objects = row[:2]
        except ValueError:
            raise(ValueError('line {}: format should be \'img_file, num_of_objects\' or \'img_file,,,,,\''.format(line)), None)

        if img_file not in result:
            result[img_file] = []

        # If a row contains only an image path, it's an image without annotations.
        if (num_of_objects) == (''):
            raise(ValueError('image {}: doesnt contain label\''.format(img_file)), None)

        # Check that the bounding box is valid.
        if int(num_of_objects) <= 0:
            raise ValueError('num_of_objects must be higher than 0 but is {}'.format(num_of_objects))

        result[img_file] = num_of_objects
    return result



def read_annotations_OC(csv_reader):
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            img_file, x, y = row[:3]
        except ValueError:
            raise(ValueError('line {}: format should be \'img_file,x,y\' or \'img_file,,,,,\''.format(line)), None)

        if img_file not in result:
            result[img_file] = []

        # If a row contains only an image path, it's an image without annotations.
        if (x, y) == ('', ''):
            raise (ValueError('image {}: doesnt contain label\''.format(img_file)), None)

        x1 = int(x)
        y1 = int(y)

        # Check that the bounding box is valid.
        if x1 < 0:
            raise ValueError('line {}: x ({}) must be higher than 0 ({})'.format(line, x))
        if y1 < 0:
            raise ValueError('line {}: y ({}) must be higher than 0 ({})'.format(line, y))

        result[img_file].append({'x': x, 'y': y})
    return result