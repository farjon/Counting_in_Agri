import os
import cv2
import warnings
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from .read_activations import get_activations

def draw_box(image, box, color, thickness=2):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.

    # Arguments
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def draw_boxes(image, boxes, color, thickness=2):
    """ Draws boxes on an image with a given color.

    # Arguments
        image     : The image to draw on.
        boxes     : A [N, 4] matrix (x1, y1, x2, y2).
        color     : The color of the boxes.
        thickness : The thickness of the lines to draw boxes with.
    """
    for b in boxes:
        draw_box(image, b, color, thickness=thickness)


def draw_detections(image, boxes, scores, labels, color=None, label_to_name=None, score_threshold=0.5):
    """ Draws detections in an image.

    # Arguments
        image           : The image to draw on.
        boxes           : A [N, 4] matrix (x1, y1, x2, y2).
        scores          : A list of N classification scores.
        labels          : A list of N labels.
        color           : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name   : (optional) Functor for mapping a label to a name.
        score_threshold : Threshold used for determining what detections to draw.
    """
    selection = np.where(scores > score_threshold)[0]

    for i in selection:
        c = color if color is not None else label_color(labels[i])
        draw_box(image, boxes[i, :], color=c)

        # draw labels
        caption = (label_to_name(labels[i]) if label_to_name else labels[i]) + ': {0:.2f}'.format(scores[i])
        draw_caption(image, boxes[i, :], caption)


def draw_annotations(image, annotations, color=(0, 255, 0), label_to_name=None):
    """ Draws annotations in an image.

    # Arguments
        image         : The image to draw on.
        annotations   : A [N, 5] matrix (x1, y1, x2, y2, label).
        color         : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name : (optional) Functor for mapping a label to a name.
    """
    for a in annotations:
        label   = a[4]
        c       = color if color is not None else label_color(label)
        caption = '{}'.format(label_to_name(label) if label_to_name else label)
        draw_caption(image, a, caption)

        draw_box(image, a, color=c)



def show_annotations_on_image(image, annotations, save_path):
    import PIL.ImageDraw as ImageDraw
    import PIL.Image as Image
    import copy
    im_drawing = Image.fromarray(np.uint8(copy.deepcopy(image)))
    draw_mask = ImageDraw.Draw(im_drawing)
    for point in annotations:
        x_tl = round(int(point[0]) - 10)
        x_br = round(int(point[0]) + 10)
        y_tl = round(int(point[1]) - 10)
        y_br = round(int(point[1]) + 10)
        draw_mask.ellipse([x_tl, y_tl, x_br, y_br], fill='blue')
    im_drawing.save(save_path)


def plot_RP_curve(recall, precision, ap, save_path):
    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})
    plt.figure()
    plt.rcParams.update({'font.size': 20})
    plt.step(recall, precision, color='b', alpha=0.99)
    plt.fill_between(recall, precision, step='post', color='b', alpha=0.1, linewidth=3)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('AP={0:0.2f}'.format(ap))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plot_path = os.path.join(save_path + '\\RP_curve.png')
    plt.savefig(plot_path)
    plt.close(plot_path)

def visualize_images(output, Image_name, save_path, generator, model, image):
    # Visualization for training or testing
    if not generator.epoch == None:
        current_epoch = str(generator.epoch+1) if generator.epoch == None else 'test'
    else:
        current_epoch = 'test'

    visualization_path = os.path.join(save_path, 'epoch_' + current_epoch)
    os.makedirs(visualization_path, exist_ok=True)

    # Draw GT activations:
    plt.figure()
    plt.axis("off")
    background = Image.open(os.path.join(generator.base_dir, 'RGB', f'{Image_name}.jpg'), 'r')
    BG_w, BG_h = background.size
    plt.imshow(background)
    plt.savefig(visualization_path + '/' + Image_name + '_BG.png', pad_inches=0)  # transparent=True,
    plt.close()

    anno = output[2][0, :, :, 0]
    plt.figure()
    #heat_map = seaborn.heatmap(anno, xticklabels=False, yticklabels=False, cbar=False)
    #heat_map = heat_map.despine
    #heat_map = heat_map.get_figure()
    plt.imshow(anno)
    plt.imsave(visualization_path + '/' + Image_name + '_anno.jpg', anno)
    gt_anns = Image.open(visualization_path + '/' + Image_name + '_anno.jpg')
    gt_anns = gt_anns.resize((BG_w, BG_h))  # Image.ANTIALIAS
    plt.imsave(visualization_path + '/' + Image_name + '_anno.jpg', gt_anns)
    plt.close()

    # out = image1 * (1.0 - alpha) + image2 * alpha
    plt.figure()
    plt.axis("off")
    alphaBlended = Image.blend(gt_anns, background, 0.6)
    plt.imshow(alphaBlended)
    plt.imsave(visualization_path + '/' + Image_name + '_Blended_GT.jpg',alphaBlended )
    plt.close()

    # Relu map #######################################################################################################
    plt.figure()
    classification_submodel_activations = get_activations(model, model_inputs=image[0], print_shape_only=False,
                                                          layer_name='pyramid_classification_relu')
    classification_submodel_activations = classification_submodel_activations[0][0, :, :, 0]

    plt.imshow(classification_submodel_activations)
    plt.imsave(visualization_path + '/' + Image_name + '_Relu.jpg', classification_submodel_activations)
    relu_anns = Image.open(visualization_path + '/' + Image_name + '_Relu.jpg')

    #relu_anns = relu_anns.convert("RGBA")
    relu_anns = relu_anns.resize((BG_w, BG_h))  # Image.ANTIALIAS
    plt.imsave(visualization_path + '/' + Image_name + '_Relu.jpg', relu_anns)
    plt.close()

    plt.figure()
    plt.axis("off")
    alphaBlended_relu = Image.blend(relu_anns, background, 0.6)
    plt.imshow(alphaBlended_relu)
    plt.imsave(visualization_path + '/' + Image_name + '_Blended_Relu.jpg', alphaBlended_relu)
    plt.close()

    # softmax map #####################################################################################################

    plt.figure()
    local_soft_max_activations = get_activations(model, model_inputs=image[0], print_shape_only=False,
                                                 layer_name='LocalSoftMax')
    local_soft_max_activations = local_soft_max_activations[0][0, :, :, 0]

    plt.imshow(local_soft_max_activations)
    plt.imsave(visualization_path + '/' + Image_name + '_softmax.png', local_soft_max_activations)
    softmax_anns = Image.open(visualization_path + '/' + Image_name + '_softmax.jpg')

    #softmax_anns = softmax_anns.convert("RGBA")
    softmax_anns = softmax_anns.resize((BG_w, BG_h))  # Image.
    plt.imsave(visualization_path + '/' + Image_name + '_softmax.jpg', softmax_anns)
    plt.close()

    plt.figure()
    plt.axis("off")
    alphaBlended_softmax = Image.blend(softmax_anns, background, 0.6)
    plt.imshow(alphaBlended_softmax)
    plt.imsave(visualization_path + '/' + Image_name + '_Blended_softmax.jpg', alphaBlended_softmax)
    plt.close()

def label_color(label):
    """ Return a color from a set of predefined colors. Contains 80 colors in total.

    Args
        label: The label to get the color for.

    Returns
        A list of three values representing a RGB color.

        If no color is defined for a certain label, the color green is returned and a warning is printed.
    """
    if label < len(colors):
        return colors[label]
    else:
        warnings.warn('Label {} has no color, returning default.'.format(label))
        return (0, 255, 0)


colors = [
    [31  , 0   , 255] ,
    [0   , 159 , 255] ,
    [255 , 95  , 0]   ,
    [255 , 19  , 0]   ,
    [255 , 0   , 0]   ,
    [255 , 38  , 0]   ,
    [0   , 255 , 25]  ,
    [255 , 0   , 133] ,
    [255 , 172 , 0]   ,
    [108 , 0   , 255] ,
    [0   , 82  , 255] ,
    [0   , 255 , 6]   ,
    [255 , 0   , 152] ,
    [223 , 0   , 255] ,
    [12  , 0   , 255] ,
    [0   , 255 , 178] ,
    [108 , 255 , 0]   ,
    [184 , 0   , 255] ,
    [255 , 0   , 76]  ,
    [146 , 255 , 0]   ,
    [51  , 0   , 255] ,
    [0   , 197 , 255] ,
    [255 , 248 , 0]   ,
    [255 , 0   , 19]  ,
    [255 , 0   , 38]  ,
    [89  , 255 , 0]   ,
    [127 , 255 , 0]   ,
    [255 , 153 , 0]   ,
    [0   , 255 , 255] ,
    [0   , 255 , 216] ,
    [0   , 255 , 121] ,
    [255 , 0   , 248] ,
    [70  , 0   , 255] ,
    [0   , 255 , 159] ,
    [0   , 216 , 255] ,
    [0   , 6   , 255] ,
    [0   , 63  , 255] ,
    [31  , 255 , 0]   ,
    [255 , 57  , 0]   ,
    [255 , 0   , 210] ,
    [0   , 255 , 102] ,
    [242 , 255 , 0]   ,
    [255 , 191 , 0]   ,
    [0   , 255 , 63]  ,
    [255 , 0   , 95]  ,
    [146 , 0   , 255] ,
    [184 , 255 , 0]   ,
    [255 , 114 , 0]   ,
    [0   , 255 , 235] ,
    [255 , 229 , 0]   ,
    [0   , 178 , 255] ,
    [255 , 0   , 114] ,
    [255 , 0   , 57]  ,
    [0   , 140 , 255] ,
    [0   , 121 , 255] ,
    [12  , 255 , 0]   ,
    [255 , 210 , 0]   ,
    [0   , 255 , 44]  ,
    [165 , 255 , 0]   ,
    [0   , 25  , 255] ,
    [0   , 255 , 140] ,
    [0   , 101 , 255] ,
    [0   , 255 , 82]  ,
    [223 , 255 , 0]   ,
    [242 , 0   , 255] ,
    [89  , 0   , 255] ,
    [165 , 0   , 255] ,
    [70  , 255 , 0]   ,
    [255 , 0   , 172] ,
    [255 , 76  , 0]   ,
    [203 , 255 , 0]   ,
    [204 , 0   , 255] ,
    [255 , 0   , 229] ,
    [255 , 133 , 0]   ,
    [127 , 0   , 255] ,
    [0   , 235 , 255] ,
    [0   , 255 , 197] ,
    [255 , 0   , 191] ,
    [0   , 44  , 255] ,
    [50  , 255 , 0]
]
