import os
import json

import pandas as pd
from PIL import Image
from glob import glob

def find_anno(annotations_for_image, box):
    '''

    :param annotations_for_image:
    :param box:
    :return:
    '''
    bbox_in_farme = {
        'x_min': [],
        'y_min': [],
        'w': [],
        'h': []
    }
    for anno in annotations_for_image:
        x_min = anno['bbox'][0]
        y_min = anno['bbox'][1]
        x_max = anno['bbox'][0] + anno['bbox'][2]
        y_max = anno['bbox'][1] + anno['bbox'][3]
        center_x = x_min + (x_max - x_min) / 2
        center_y = y_min + (y_max - y_min) / 2
        if center_x > box[0] and center_x < box[2] and center_y > box[1] and center_y < box[3]:
            if x_min > box[0]:
                bbox_x_min = x_min - box[0]
            else:
                bbox_x_min = 1
            if y_min > box[1]:
                bbox_y_min = y_min - box[1]
            else:
                bbox_y_min = 1
            if x_max < box[2]:
                bbox_x_max = x_max - box[0]
            else:
                bbox_x_max = box[2] - 1 - box[0]
            if y_max < box[3]:
                bbox_y_max = y_max - box[1]
            else:
                bbox_y_max = box[3] - 1 - box[1]
            bbox_in_farme['x_min'].append(bbox_x_min)
            bbox_in_farme['y_min'].append(bbox_y_min)
            bbox_in_farme['w'].append(bbox_x_max - bbox_x_min)
            bbox_in_farme['h'].append(bbox_y_max - bbox_y_min)

    return bbox_in_farme

def split_to_tiles(args, tiles=10, pad=30, MSR_DRN=False):
    """
    This function transform the raw images and annotations to tiles based on coco format annotations
    and creates all three annotation formats - detection (coco format), direct regression (csv format)
    if MSR_DRN is True, it will create the annotations for the MSR_DRN model
    :param args:
    :param tiles:
    :param pad:
    :param MSR_DRN:
    :return:
    """
    set_to_split = ['train', 'test', 'val']
    output_dir = os.path.join(args.ROOT_DIR, 'Data', args.data + '_split')
    os.makedirs(output_dir, exist_ok=True)
    for current_set in set_to_split:
        # input information
        to_tile_dir = os.path.join(args.ROOT_DIR, 'Data', args.data, 'Detection', 'coco', current_set)
        to_tile_annotations_file = os.path.join(args.ROOT_DIR, 'Data', args.data, 'Detection', 'coco', 'annotations', 'instances_' + current_set + '.json')
        # coco output information
        det_imgs_output_dir = os.path.join(output_dir, 'Detection', 'coco', current_set)
        det_output_anno_dir = os.path.join(output_dir, 'Detection', 'coco', 'annotations')
        det_output_anno_file = os.path.join(det_output_anno_dir, 'instances_' + current_set + '.json')
        # direct regression output information
        count_output_imgs_dir = os.path.join(output_dir, 'Direct_regression', current_set)
        count_output_anno_dir = os.path.join(output_dir, 'Direct_regression', 'annotations')
        count_output_anno_file = os.path.join(count_output_anno_dir, current_set + '.csv')
        # MSR_DRN output information
        if MSR_DRN:
            msr_drn_output_anno_dir = os.path.join(output_dir, 'MSR_DRN', current_set)
            os.makedirs(msr_drn_output_anno_dir, exist_ok=True)

        os.makedirs(det_imgs_output_dir, exist_ok=True)
        os.makedirs(det_output_anno_dir, exist_ok=True)
        os.makedirs(count_output_imgs_dir, exist_ok=True)
        os.makedirs(count_output_anno_dir, exist_ok=True)

        # coco format
        categories = [{'supercategory': None, 'id': 1, 'name': args.data}]
        images = []
        annotations = []
        image_id = 0
        annotation_id = 0

        # count format
        collect_count_annotations = {
            'image_name': [],
            'count': []
        }

        anno_file = json.load(open(to_tile_annotations_file))
        for image_desc in anno_file['images']:
            image_name = image_desc['file_name']
            annotations_for_image = [x for x in anno_file['annotations'] if x['image_id'] == image_desc['id']]
            print(f'working on image {image_name}')
            sub_image_counter = 0
            im = Image.open(os.path.join(to_tile_dir, image_name))
            im_widht, im_height = im.size
            h_step, w_step = im_height//tiles, im_widht//tiles
            end_of_width = im_widht // w_step - 1
            end_of_height = im_height // h_step - 1
            for i in range(im_height // h_step):
                for j in range(im_widht // w_step):
                    sub_image_name = image_name.split('.')[0] + "_" + str(sub_image_counter) +'.jpg'
                    sub_image_counter += 1
                    if j < end_of_width:
                        w_s = 0
                        w_e = pad * (im_widht / im_height) # keeping image w/h ratio
                    else:
                        w_s = pad * (im_widht / im_height) # keeping image w/h ratio
                        w_e = 0
                    if i < end_of_height:
                        h_s = 0
                        h_e = pad
                    else:
                        h_s = pad
                        h_e = 0
                    box = (j * w_step - w_s, i * h_step - h_s,
                           (j + 1) * w_step + w_e, (i + 1) * h_step + h_e)
                    # cropping the sub image
                    cropped_frame = im.crop(box)
                    # collecting annotations for sub image
                    bbox_in_cropped_frame = find_anno(annotations_for_image, box)
                    # in case that the image has no object - skip it
                    if len(bbox_in_cropped_frame['x_min']) == 0:
                        continue
                    # saving cropped image
                    cropped_frame.save(os.path.join(det_imgs_output_dir, sub_image_name))
                    cropped_frame.save(os.path.join(count_output_imgs_dir, sub_image_name))

                    # collect coco format annotations
                    sub_image_w = cropped_frame.size[0]
                    sub_image_h = cropped_frame.size[1]
                    images.append({
                        'file_name': sub_image_name, 'height': sub_image_h, 'width': sub_image_w,
                        'id': image_id
                    })
                    for k in range(len(bbox_in_cropped_frame['x_min'])):
                        bbox = [bbox_in_cropped_frame['x_min'][k],
                                bbox_in_cropped_frame['y_min'][k],
                                bbox_in_cropped_frame['w'][k],
                                bbox_in_cropped_frame['h'][k]]
                        annotations.append({'bbox': bbox,
                                            'category_id': 1,
                                            'area': int(bbox[2] * bbox[3]),
                                            'id': annotation_id,
                                            'image_id': image_id,
                                            'iscrowd': 0,
                                            'ignore': 0})
                        annotation_id += 1

                    # collect count format annotations
                    collect_count_annotations['image_name'].append(sub_image_name)
                    collect_count_annotations['count'].append(len(bbox_in_cropped_frame['x_min']))

                    image_id += 1
        # dump coco format annotations
        json_dict = {
            'categories': categories,
            'images': images,
            'annotations': annotations
        }
        with open(det_output_anno_file, "w") as outfile:
            json.dump(json_dict, outfile)
        with open(os.path.join(det_imgs_output_dir, f'instances_{current_set}.json'), "w") as outfile:
            json.dump(json_dict, outfile)
        # dump count format annotations
        pd.DataFrame(collect_count_annotations).to_csv(count_output_anno_file, index=False)
    print('finish splitting the images for train, test, and val directories')