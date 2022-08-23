import os
import pandas as pd
from PIL import Image
from glob import glob

def find_anno(anno_file, box):
    bbox_in_farme = {
        'x_min': [],
        'y_min': [],
        'x_max': [],
        'y_max': []
    }
    for i, row in anno_file.iterrows():
        x_min = row.loc['xmin']
        y_min = row.loc['ymin']
        x_max = row.loc['xmax']
        y_max = row.loc['ymax']
        center_x = x_min + (x_max - x_min) / 2
        center_y = y_min + (y_max - y_min) / 2
        if center_x > box[0] and center_x < box[2] and center_y > box[1] and center_y < box[3]:
            bbox_in_farme['x_min'].append(x_min)
            bbox_in_farme['y_min'].append(y_min)
            bbox_in_farme['x_max'].append(x_max)
            bbox_in_farme['y_max'].append(y_max)

    return pd.DataFrame(bbox_in_farme)


def split_to_tiles(args, tiles=10, pad=30):
    set_to_split = ['train', 'test', 'val']
    output_dir = os.path.join(args.ROOT_DIR, 'Data', args.data + '_split')
    print('Notice - the files should be in a .jpg format')
    if os.path.exists(output_dir):
        print('There is already a splitted dataset, if you wish to use different tile splits, '
              'delete the directory and re-run the train,'
              'moving forward with the training process')
        return
    for current_set in set_to_split:
        to_tile_dir = os.path.join(args.ROOT_DIR, 'Data', args.data, 'coco', current_set)
        imgs_output_dir = os.path.join(output_dir, 'coco', current_set)
        det_output_anno_dir = os.path.join(output_dir, 'annotations', current_set)
        count_output_anno_dir = os.path.join(output_dir, 'regression annotations')
        os.makedirs(imgs_output_dir, exist_ok=True)
        os.makedirs(det_output_anno_dir, exist_ok=True)
        os.makedirs(count_output_anno_dir, exist_ok=True)
        sub_images_csv = {
            'image_name': [],
            'GT_number': []
        }
        image_format = glob(os.path.join(to_tile_dir, '*'))[0][-3:]
        anno_file = pd.read_csv(os.path.join(args.ROOT_DIR, 'Data', args.data, 'coco', 'annotations', 'instances_' + current_set + '.csv'))
        for k, file_name in enumerate(glob(os.path.join(to_tile_dir, '*.'+image_format))):
            image_name = file_name.split('\\')[-1].split('.')[0] # '/' to split the path, '.' to split the ending
            # anno_file = pd.read_csv(os.path.join(args.ROOT_DIR, 'Data', args.data, 'annotations', current_set, image_name + '.csv'))
            print(f'working on image {image_name}')
            sub_image_counter = 0
            im = Image.open(file_name)
            im_widht, im_height = im.size
            h_step, w_step = im_height//tiles, im_widht//tiles
            end_of_width = im_widht // w_step - 1
            end_of_height = im_height // h_step - 1
            for i in range(im_height // h_step):
                for j in range(im_widht // w_step):
                    sub_image_name = image_name + "_" + str(sub_image_counter)
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
                    bbox_in_cropped_frame = find_anno(anno_file, box)
                    # in case that the image has no object - skip it
                    if bbox_in_cropped_frame.shape[0] == 0:
                        continue
                    # saving cropped image
                    cropped_frame.save(os.path.join(imgs_output_dir, sub_image_name + '.jpg'))
                    # saving sub image detection annotations
                    bbox_in_cropped_frame.to_csv(os.path.join(det_output_anno_dir, sub_image_name + '.csv'), index=False)
                    # collecting sub images number of objects
                    sub_images_csv['image_name'].append(sub_image_name + '.jpg')
                    sub_images_csv['GT_number'].append(bbox_in_cropped_frame.shape[0])

        pd.DataFrame(sub_images_csv).to_csv(os.path.join(count_output_anno_dir, current_set + '.csv'), index=False)

    print('finish splitting the images for train, test, and val directories')