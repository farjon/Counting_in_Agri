import os
import json
import argparse
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import copy

def parse_args():
    parser = argparse.ArgumentParser(description='Basic regression pipe using a deep neural network.')
    # --------------------------- Data Arguments ---------------------------
    parser.add_argument('-d', '--data', type=str, default='AppleFlowers_split', help='choose a dataset')
    parser.add_argument('-o', '--output_path', type=str, default='', help='path to save images, defualt is NONE')
    args = parser.parse_args()
    return args

def drawrect(drawcontext, xy, outline=None, width=0):
    (x1, y1), (x2, y2) = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)

def main(args):
    args.data_path = os.path.join(args.ROOT_DIR, 'Data', args.data, 'coco')
    args.output_path = os.path.join(args.data_path, 'images_with_annotations')
    os.makedirs(args.output_path, exist_ok=True)
    sets = ['train', 'val', 'test']
    for current_set in sets:
        annotations_file_path = os.path.join(args.data_path, 'annotations', 'instances_' + current_set + '.json')
        json_decoded = json.load(open(annotations_file_path))
        for image_desc in json_decoded['images']:
            im_original = Image.open(os.path.join(args.data_path, current_set, image_desc['file_name']))
            im_draw = copy.deepcopy(im_original)
            draw_mask = ImageDraw.Draw(im_draw)
            annotations_for_image = [x for x in json_decoded['annotations'] if x['image_id'] == image_desc['id']]

            for anno in annotations_for_image:
                bbox = anno['bbox']
                x = bbox[0]
                y = bbox[1]
                w = bbox[2]
                h = bbox[3]

                drawrect(draw_mask, [(x, y), (x + w, y + h)], outline="red", width=5)

            new_file_name = os.path.join(args.output_path, image_desc['file_name'])
            save_image = Image.new('RGB', im_original.size)
            save_image.paste(im_draw)
            save_image.save(new_file_name)

if __name__ == '__main__':
    args = parse_args()
    args.ROOT_DIR = 'C:\\Users\\owner\\PycharmProjects\\Counting_in_Agri'
    main(args)