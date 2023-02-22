import torch
import numpy as np
from tqdm import tqdm

def test_model(args, test_dataset, model):
    device = args.device
    model_counting_results = {
        'image_name': [],
        'gt_count': [],
        'pred_count': []
    }

    model.to(device)
    model.eval()
    test_loop = tqdm(enumerate(test_dataset), total=len(test_dataset), leave=False)
    for test_batch_idx, (inputs, labels, img_details) in test_loop:
        with torch.no_grad():
            inputs = inputs.to(device)
            outputs = model(inputs)

            model_counting_results['image_name'].extend(img_details)
            model_counting_results['gt_count'].extend(labels.numpy().tolist())
            model_counting_results['pred_count'].extend(outputs.detach().cpu().numpy().tolist())


    return model_counting_results