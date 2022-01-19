import torch
import wandb
import numpy as np
from tqdm import tqdm

def test_models(args, test_dataset, loss_func, models):
    device = args.device
    models_scores = []
    for model in models:
        model.to(device)
        model.eval()
        test_loss = []
        test_loop = tqdm(enumerate(test_dataset), total=len(test_dataset), leave=False)
        for test_batch_idx, (inputs, labels) in test_loop:
            with torch.no_grad():
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_func(outputs, labels)
                test_loss.append(loss.item())

        mean_test_loss = np.mean(test_loss)

        print('Test. model - {}. loss: {:1.5f}.'.format(
                model, mean_test_loss))
        models_scores.append(mean_test_loss)
    wandb.log({'final_MSE':models_scores[0]})
    wandb.log({'best_MSE':models_scores[1]})
    return models_scores