from ..utils.image import (
    TransformParameters,
    adjust_transform_for_image,
    apply_transform,
    preprocess_image,
    resize_image,
    read_image_bgr
)
from ..utils.transform import transform_ab
import keras
import numpy as np
from PIL import Image
from six import raise_from
import random
import threading
import csv
import os.path
import cv2

def images_ratios(image_shape, output_shape):
    return output_shape / np.array(image_shape[:2])


class CSVGenerator_MSR_DRN(object):
    def __init__(
        self,
        mode='training',
        model_type='DRN',
        class_name='',
        csv_object_number_file = '',
        csv_object_location_file = '',
        base_dir=None,
        batch_size=1,
        epoch=None,
        transform_generator=None,
        image_min_side=800,
        image_max_side=1333,
        transform_parameters=None,
        group_method='random',  # one of 'none', 'random', 'ratio'
        shuffle_groups=False,
        **kwargs
    ):
        self.mode = mode  # training or inference
        if mode == 'training':
            assert csv_object_number_file != '', 'if in training mode, number of objects should be provided'
            self.csv_object_number_file = csv_object_number_file
            if model_type == 'DRN':
                assert csv_object_location_file != '', 'if in training mode and using the DRN, objects dot annotations should be provided'
                self.csv_object_location_file = csv_object_location_file

        self.epoch       = epoch
        self.model_type = model_type
        self.class_name = class_name

        # Take base_dir from annotations file if not explicitly specified.
        self.base_dir = base_dir
        if self.base_dir is None:
            self.base_dir = os.path.dirname(csv_object_number_file)

        self.transform_generator    = transform_generator
        self.batch_size             = int(batch_size)
        self.group_method           = group_method
        self.shuffle_groups         = shuffle_groups
        self.image_min_side         = image_min_side
        self.image_max_side         = image_max_side
        self.transform_parameters   = transform_parameters or TransformParameters()
        self.group_index            = 0
        self.lock                   = threading.Lock()

        if mode == 'training':
            # csv with img_path, num_of_objects
            try:
                with open(csv_object_number_file, 'r', newline='') as file:
                    self.image_data_object_number = self._read_number_of_objects(csv.reader(file, delimiter=','))
            except ValueError as e:
                raise_from(ValueError(f'invalid CSV annotations file: {csv_object_number_file}: {e}'), None)

            self.rbg_images_names = list(self.image_data_object_number.keys())
            # csv with img_path, x, y
            if model_type == 'DRN':
                try:
                    with open(csv_object_location_file, 'r', newline='') as file:
                        self.image_data_object_location = self._read_annotations_objects_locations(csv.reader(file, delimiter=','))
                except ValueError as e:
                    raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(csv_object_location_file, e)), None)

                # make sure all files has center annotations
                self.centers_images_names = [f'{x}_centers' for x in self.rbg_images_names]
                assert len(list(self.image_data_object_location.keys())) == len(self.centers_images_names) , 'there are some missing centers annotations'
        else:
            rbg_images_names = os.listdir(self.base_dir)
            rbg_images_names_a =[]

            for im in rbg_images_names:
                if 'CVPPP' in base_dir:
                    if im.split('_')[-1] == 'rgb.png':
                        rbg_images_names_a.append(im)
                else:
                    rbg_images_names_a.append(im)
            self.rbg_images_names = rbg_images_names_a

        self.group_images()
    # read the number of objects (if in training)
    def _read_number_of_objects(self, csv_reader):
        result = {}
        for line, row in enumerate(csv_reader):
            try:
                img_file, num_of_objects = row[:2]
            except ValueError:
                raise_from(ValueError(
                    'line {}: format should be \'img_file, num_of_objects\' or \'img_file,\''.format(line + 1)), None)

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (num_of_objects) == (''):
                raise (ValueError('image {}: doesnt contain label\''.format(img_file)), None)

            # Check that the bounding box is valid.
            if int(num_of_objects) <= 0:
                raise ValueError('num_of_leafs must be higher than 0 but is {}'.format(num_of_objects))

            result[img_file].append({'num_of_objects': num_of_objects, 'class': self.class_name})
        return result
    # read object locations (if using the DRN in training)
    def _read_annotations_objects_locations(self, csv_reader):
        result = {}
        for line, row in enumerate(csv_reader):
            try:
                img_file, x, y = row[:3]
            except ValueError:
                raise_from(ValueError('line {}: format should be \'img_file,x,y\' or \'img_file,\''.format(line + 1)),
                           None)

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

            result[img_file].append({'x': x, 'y': y, 'class': self.class_name})
        return result
    # create an order of reading the images
    def group_images(self):
        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    def get_epoch(self):
        return self.epoch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def size(self):
        return len(self.rbg_images_names)

    def image_path_rgb(self, image_index):
        return self.find_image_path(self.rbg_images_names[image_index], 'RGB')


    def image_path_centers(self, image_index):
        return self.find_image_path(self.centers_images_names[image_index], 'centers')


    def find_image_path(self, image_name, dir_name):
        if '.' in os.path.splitext(image_name)[-1]:
            return os.path.join(self.base_dir, image_name)
        else:
            try_format = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
            for f in try_format:
                if os.path.exists(os.path.join(self.base_dir, dir_name, image_name + '.' + f)):
                    return os.path.join(self.base_dir, dir_name, image_name + '.' + f)

    def image_aspect_ratio(self, image_index):
        # PIL is fast for metadata
        image = Image.open(self.image_path_rgb(image_index))
        return float(image.width) / float(image.height)

    def load_image_rgb(self, image_index):
        return read_image_bgr(self.image_path_rgb(image_index))

    def load_image_centers(self, image_index):
        return read_image_bgr(self.image_path_centers(image_index))

    def load_image_byName(self, image_name):
        return read_image_bgr(os.path.join(self.base_dir, image_name))

    def load_annotations_num_of_objects(self, image_index):
        path   = self.rbg_images_names[image_index]
        annots = self.image_data_object_number[path]
        counts  = np.zeros((len(annots), 2))

        for idx, annot in enumerate(annots):
            counts[idx, 0] = float(annot['num_of_objects'])
            counts[idx, 1] = 0

        return counts

    def load_annotations_group_num_of_objects(self, group):
        return [self.load_annotations_num_of_objects(image_index) for image_index in group]

    def load_annotations_objects_centers(self, image_index):
        path   = self.rbg_images_names[image_index]
        annots = self.image_data_object_location[path]
        centers  = np.zeros((len(annots), 3))

        for idx, annot in enumerate(annots):
            centers[idx, 0] = float(annot['x'])
            centers[idx, 1] = float(annot['y'])
            centers[idx, 2] = 0

        return centers

    def load_annotations_group_objects_center(self, group):
        return [[self.load_annotations_objects_centers(image_index) for image_index in group]]

    def load_image_group(self, group):
        return [self.load_image_rgb(image_index) for image_index in group]

    def load_centers_image_group(self, group):
        return [self.load_image_centers(image_index) for image_index in group]

    def random_transform_rgb_image(self, rgb_image):
        # this generator operates for both training and testing
        if self.transform_generator:
            transformation_to_apply = next(self.transform_generator)
            transform_rgb = adjust_transform_for_image(transformation_to_apply, rgb_image,
                                                       self.transform_parameters.relative_translation)
            res_rgb_image = apply_transform(transform_rgb, rgb_image, self.transform_parameters)
            return res_rgb_image
        else:
            return rgb_image
    def random_transform_rbg_centers_images(self, rgb_image, annotations):
        random_success_flag = False

        # randomly transform both image and annotations
        # this generator operates for both training and testing
        if self.transform_generator:
            while(random_success_flag == False):
                transformation_to_apply = next(self.transform_generator)
                transform_rgb = adjust_transform_for_image(transformation_to_apply, rgb_image, self.transform_parameters.relative_translation)
                res_rgb_image     = apply_transform(transform_rgb, rgb_image, self.transform_parameters)
                annotations_a = annotations.copy()
                for index in range(annotations_a.shape[0]):
                    annotations_a[index, :2] = transform_ab(transform_rgb, annotations_a[index, :2])
                random_success_flag = True
                assert annotations.shape[0]==annotations_a.shape[0], 'there is some buge in the augmantation procses'
                # check if annotations_a contains negative values
                for row in annotations_a:
                    for ele in row:
                        if ele < 0:
                            random_success_flag = False
            return res_rgb_image, annotations_a
        else:
            return rgb_image, annotations

    def resize_image(self, image):
        return resize_image(image)#, min_side=self.image_min_side, max_side=self.image_max_side)

    def resize_map(self, map, scale):
        return cv2.resize(map, None, fx=scale, fy=scale)

    def preprocess_image(self, image):
        return preprocess_image(image)

    def compute_keypoints_targets(self, image_shape, annotations_leaves_centers,):
        pyramid_level = 3
        output_shape = (np.array(image_shape[:2]) + 2 ** pyramid_level - 1) // (2 ** pyramid_level)
        image_ratio = images_ratios(image_shape, output_shape)
        annotations_leaves_centers[:, :2] = annotations_leaves_centers[:, :2] * image_ratio

        annotations = np.zeros(output_shape)
        for i in range(annotations_leaves_centers.shape[0]):

            gaussian_map = create_gausian_mask(annotations_leaves_centers[i, :2], output_shape[1],output_shape[0])
            # each center point in the GT will be 1 in the annotation map
            annotations = np.maximum(annotations, gaussian_map)
        #assert len(np.where(annotations==1)[0])==annotations_leaves_centers.shape[0], 'there is some bug in the gaussian creation procsess
        return annotations

    # def get_annotations_byName(self):
    #     annotations_dict = {}
    #     with open(self.csv_object_number_file) as csvfile:
    #         readCSV = csv.reader(csvfile)
    #         for row in readCSV:
    #             annotations_dict[row[0]] = int(row[1])
    #     return (annotations_dict)

    # def get_object_coord_byName(self):
    #     coordinates_dict = {}
    #     with open(self.csv_object_location_file) as csvfile:
    #         readCSV = csv.reader(csvfile)
    #         for row in readCSV:
    #             if len(coordinates_dict) == 0:
    #                 coordinates_dict[row[0]] = []
    #                 coordinates_dict[row[0]].append([int(row[1]),int(row[2])])
    #             else:
    #                 if row[0] in coordinates_dict.keys():
    #                     coordinates_dict[row[0]].append([int(row[1]), int(row[2])])
    #                 else:
    #                     coordinates_dict[row[0]]=[]
    #                     coordinates_dict[row[0]].append([int(row[1]), int(row[2])])
    #
    #     return (coordinates_dict)

    def preprocess_group_entry_MSR(self, image):
        # preprocess the image - based on keras image processing
        image = self.preprocess_image(image)
        # resize image
        image, image_scale = self.resize_image(image)
        # randomly transform image and annotations
        image_transeformed = self.random_transform_rgb_image(image)

        return image_transeformed


    def preprocess_group_MSR(self, image_group):

        for index, image in enumerate(image_group):
            # preprocess a single group entry
            image = self.preprocess_group_entry_MSR(image)
            # copy processed data back to group
            image_group[index] = image

        return image_group

    def preprocess_group_DRN(self, image_group, annotations_group_objects_center):

        for index, (image, annotations) in enumerate(zip(image_group, annotations_group_objects_center[0])):
            # preprocess a single group entry
            image, annotations = self.preprocess_group_entry_DRN(image, annotations)
            # copy processed data back to group
            image_group[index]       = image
            annotations_group_objects_center[0][index] = annotations

        return image_group, annotations_group_objects_center

    def preprocess_group_entry_DRN(self, image, annotations):
        # preprocess the image
        image = self.preprocess_image(image)
        # resize image
        image, image_scale = self.resize_image(image)
        # apply resizing to annotations too
        annotations[:, :2] *= image_scale
        # randomly transform image and annotations
        image_transeformed, annotation_transeformed = self.random_transform_rbg_centers_images(image, annotations)

        # create different sizes of masks to guide the network
        annotation_map_r_3 = compute_keypoints_targets_DRN(image_transeformed.shape, annotation_transeformed, radius = (3,3))
        annotation_map_r_7 = compute_keypoints_targets_DRN(image_transeformed.shape, annotation_transeformed, radius = (7,7))
        annotation_map_r_5 = compute_keypoints_targets_DRN(image_transeformed.shape, annotation_transeformed, radius = (5,5))

        # the order here is the same order as the output of the model
        # [mid_out_0, mid_out_1, mid_out_2, mid_out_3, final_relu]
        annotation_map = [annotation_map_r_7,annotation_map_r_5,annotation_map_r_5,annotation_map_r_3,annotation_map_r_3]

        return image_transeformed, annotation_map


    def preprocess_group_input(self, image_group):
        for index, image in enumerate(image_group):
            # preprocess a single group entry
            # preprocess the image
            image = self.preprocess_image(image)
            # resize image
            image,_ = self.resize_image(image)

            # copy processed data back to group
            image_group[index]       = image

        return image_group

    def compute_inputs(self, image_group):
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        return image_batch

    def compute_targets_MSR(self, image_group, annotations_group_num_of_objects):

        # compute regression targets
        regression_group = [None] * self.batch_size
        for index, (image, annotations_num_of_objects) in enumerate(zip(image_group, annotations_group_num_of_objects)):
            regression_group[index] = annotations_num_of_objects[0][0]

        regression_batch = np.zeros((self.batch_size,) + regression_group[0].shape, dtype=keras.backend.floatx())

        # copy all labels and regression values to the batch blob
        for index, regression in enumerate(regression_group):
            regression_batch[index, ...] = regression

        if self.model_type == 'MSR_P3_L2':
            ret = [regression_batch]
        elif self.model_type == 'MSR_P3_P7_Gauss_MLE':
            ret = [regression_batch] * 5

        return ret

    def compute_targets_DRN(self, image_group, annotations_group_objects_center, annotations_group_num_of_objects):

        # compute labels and regression targets
        labels_group = [None] * self.batch_size
        regression_group = [None] * self.batch_size
        for index, (image, annotations_objects_centers, annotations_num_of_objects) in enumerate(
                zip(image_group, annotations_group_objects_center, annotations_group_num_of_objects)):
            # compute regression targets
            labels_group[index] = annotations_objects_centers[0]
            regression_group[index] = annotations_num_of_objects[0][0]

        labels_batch_1 = np.zeros((self.batch_size,) + labels_group[0][0].shape, dtype=keras.backend.floatx())
        labels_batch_2 = np.zeros((self.batch_size,) + labels_group[0][0].shape, dtype=keras.backend.floatx())
        labels_batch_3 = np.zeros((self.batch_size,) + labels_group[0][0].shape, dtype=keras.backend.floatx())
        labels_batch_4 = np.zeros((self.batch_size,) + labels_group[0][0].shape, dtype=keras.backend.floatx())
        labels_batch_5 = np.zeros((self.batch_size,) + labels_group[0][0].shape, dtype=keras.backend.floatx())

        regression_batch = np.zeros((self.batch_size,) + regression_group[0].shape, dtype=keras.backend.floatx())

        # copy all labels and regression values to the batch blob
        for index, (labels, regression) in enumerate(zip(labels_group, regression_group)):
            labels_batch_1[index, ...] = labels[0]
            labels_batch_2[index, ...] = labels[1]
            labels_batch_3[index, ...] = labels[2]
            labels_batch_4[index, ...] = labels[3]
            labels_batch_5[index, ...] = labels[4]
            regression_batch[index, ...] = regression


        ret = [regression_batch, np.expand_dims(labels_batch_1, 3),
                                 np.expand_dims(labels_batch_2, 3),
                                 np.expand_dims(labels_batch_3, 3),
                                 np.expand_dims(labels_batch_4, 3),
                                 np.expand_dims(labels_batch_5, 3)]
        return ret

    def compute_input_output(self, group):
        # load images and annotations
        image_group = self.load_image_group(group)

        if self.mode == 'training':
            annotations_group_num_of_objects = self.load_annotations_group_num_of_objects(group)
            # perform preprocessing steps
            if self.model_type == 'DRN':
                annotations_group_objects_center = self.load_annotations_group_objects_center(group)
                # in this option each sub-network layer will have a different gaussian maps as targets
                image_group, annotations_group_objects_center = self.preprocess_group_DRN(
                    image_group, annotations_group_objects_center)
            else:
                image_group = self.preprocess_group_MSR(image_group)

            # compute network inputs
            inputs = self.compute_inputs(image_group)
            if self.model_type == 'DRN':
                targets = self.compute_targets_DRN(image_group, annotations_group_objects_center, annotations_group_num_of_objects)
            else:
                targets = self.compute_targets_MSR(image_group, annotations_group_num_of_objects)
            return inputs, targets

        else:
            # if in inference mode - targets aren't available
            image_group = self.preprocess_group_input(image_group)
            inputs = self.compute_inputs(image_group)
            return inputs


    def __next__(self):
        return self.next()

    def next(self):
        # advance the group index
        with self.lock:
            if self.group_index == 0 and self.shuffle_groups:
                # shuffle groups at start of epoch
                #random.shuffle(self.groups)
                pass
            group = self.groups[self.group_index]
            self.group_index = (self.group_index + 1) % len(self.groups)

        return self.compute_input_output(group)


# the following functions are relevant for DRN or MSR
def create_gausian_mask(center_point, nCols, nRows, q = 99, radius = (5,5)):

    s = 3
    if (s>=radius[0]):
        s=1
    x = np.tile(range(nCols), (nRows,1))
    y = np.tile(np.reshape(range(nRows),(nRows,1)),(1,nCols))

    x2 = (((x - round(center_point[0]))*s) / radius[0]) ** 2
    y2 = (((y - round(center_point[1]))*s) / radius[1]) ** 2

    p = np.exp(-0.5 * (x2 + y2))

    p[np.where(p < np.percentile(p, q))] = 0

    p = p / np.max(p)
    if not np.isfinite(p).all() or not np.isfinite(p).all():
        print('divide by zero')
    return p

def compute_keypoints_targets_DRN(image_shape, annotations_objects_centers ,radius=(5,5)):
    import copy
    annotations_leaves_centers = copy.deepcopy(annotations_objects_centers)
    pyramid_level = 3
    output_shape = (np.array(image_shape[:2]) + 2 ** pyramid_level - 1) // (2 ** pyramid_level)
    image_ratio = output_shape / np.array(image_shape[:2])
    annotations_leaves_centers[:, :2] = annotations_leaves_centers[:, :2] * image_ratio
    annotations = np.zeros(output_shape)
    for i in range(annotations_leaves_centers.shape[0]):
        gaussian_map = create_gausian_mask(annotations_leaves_centers[i, :2], output_shape[1],output_shape[0], radius=radius)
        # each center point in the GT will be 1 in the annotation map
        annotations = np.maximum(annotations, gaussian_map)

    if np.isnan(annotations).any():
        raise("nan was found")
    return annotations