def combine_predictions_to_full_scale(original_image_name, predictions, original_image_size, tiles):
    """
    This function combines the predictions of the tiles to the full scale image
    :param predictions:
    :param image_size:
    :param tiles:
    :return:
    """
    full_scale_predictions = []
    sub_image_counter = 0
    im_widht, im_height = original_image_size
    h_step, w_step = im_height // tiles, im_widht // tiles

    for i in range(im_height // h_step):
        for j in range(im_widht // w_step):
            sub_image_name = original_image_name.split('.')[0] + "_" + str(sub_image_counter) + '.jpg'
            sub_image_counter += 1
            if sub_image_name in predictions.keys():
                for prediction, score in zip(predictions[sub_image_name]['det'], predictions[sub_image_name]['scores']):
                    x_min = prediction[0] + j * w_step
                    y_min = prediction[1] + i * h_step
                    # format is [score, x, y, w, h]
                    full_scale_predictions.append([score, x_min, y_min, prediction[2], prediction[3]])

    return full_scale_predictions