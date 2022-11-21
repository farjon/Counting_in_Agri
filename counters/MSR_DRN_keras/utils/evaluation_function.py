from __future__ import print_function

import os
import numpy as np
from counters.MSR_DRN_keras.bin.eval_detection import detection_evaluation, calc_recall_precision_ap
from counters.MSR_DRN_keras.utils.visualization import plot_RP_curve, visualize_images


def collect_gt_and_preds(model_type, generator, model, save_path=None, visualize_im = False, calc_det_performance = False):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_GT_counts[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        model_type          : one of MSR or DRN
        generator           : The generator used to run images through the model.
        model               : The model to run on the images.
        save_path           : The path to save the images with visualized detections to.
        visualize_im        : flag to visualization
        calc_det_performance: if DRN model type, choose if you what to claculate detection performance based on PCK metric
    """

    gt_counts = []
    pred_counts = []
    # setup params for calculation of detection performance based on the PCK metric
    alpha = 0.1
    T, P = [], []
    for index in range(generator.size()):
        image, gt = generator.next()

        image_index = generator.groups[generator.group_index-1][0]
        full_rgbImage_name = generator.rbg_images_names[image_index]
        Image_name = full_rgbImage_name.split("_rgb")[0]

        if visualize_im:
            if not generator.epoch == None:
                if generator.epoch==0 or (generator.epoch+1) % 20 == 0 :
                    visualize_images(gt, Image_name, save_path, generator, model, image)
            else:
                visualize_images(gt, Image_name, save_path, generator, model, image)

        # get GT
        gt = gt[0][0]
        gt_counts.append(gt)

        if model_type == 'DRN':
            count = model.predict_on_batch(image)[0][0][0]

            print('image:', Image_name, 'GT:', gt, ', predicted:', round(count), "(", count, "),",
                  "abs_DiC:", round(abs(gt - count), 2))

            if calc_det_performance:
                t, p = detection_evaluation(os.path.join(generator.base_dir, Image_name), model, image,
                                            gt[-1][0, :, :, 0], alpha)
                T = T + t
                P = P + p

        elif model_type == 'MSR_P3_L2':
            count = model.predict_on_batch(image)[0][0]
        elif model_type == 'MSR_P3_P7_Gauss_MLE':
            count = model.predict_on_batch(image)[0]
        else:
            raise (f'Unknown model type {model_type}')

        pred_counts.append(round(count))

    if model_type == 'DRN' and calc_det_performance:
        recall, precision, ap = calc_recall_precision_ap(T, P)
        plot_RP_curve(recall, precision, ap, save_path)
        return gt_counts, pred_counts, ap

    return gt_counts, pred_counts


def collect_predictions(model_type, generator, model):

    predicted_counts = []
    for index in range(generator.size()):
        image = generator.next()
        image_index = generator.groups[generator.group_index-1][0]
        full_rgbImage_name = generator.rbg_images_names[image_index]
        Image_name = full_rgbImage_name.split("_rgb")[0]

        if model_type == 'DRN':
            count = model.predict_on_batch(image)[0][0][0]
        elif model_type == 'MSR_P3_L2':
            count = model.predict_on_batch(image)[0][0]
        elif model_type == 'MSR_P3_P7_Gauss_MLE':
            count = model.predict_on_batch(image)[0][0]

        print('image:', Image_name, ', predicted:', round(count), ", (", count, ")")
        predicted_counts.append({'image': Image_name, 'predicted': round(count)})

    return predicted_counts


def evaluate(model_type, generator, model, save_path=None, visualize_im=False, calc_det_performance = False):
        results = collect_gt_and_preds(
            model_type,
            generator,
            model,
            save_path=save_path,
            visualize_im=visualize_im,
            calc_det_performance=calc_det_performance
        )
        gt_counts, pred_counts = np.array(results[0]), np.array(results[1])

        num_of_images = len(results[0])
        DiC = np.mean(gt_counts - pred_counts)
        abs_DiC = np.mean(np.abs(gt_counts - pred_counts))
        MSE = np.mean((gt_counts - pred_counts)**2)
        agreement = (num_of_images - np.count_nonzero(gt_counts - pred_counts))/num_of_images

        if calc_det_performance:
            return DiC, abs_DiC, agreement , MSE, results[-1]

        return DiC, abs_DiC, agreement, MSE


