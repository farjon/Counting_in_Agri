import os
import numpy as np
import torch
import wandb
from tqdm import tqdm
from counters.MSR_DRN.utils.losses import focal_DRN

def train_MSR_DRN(args, train_dataset, val_dataset, model, count_loss_func, optimizer, scheduler):
    # wandb.init('DRN - counting')
    #
    # config = wandb.config
    # wandb.watch(model)

    model.to(args.device)
    model.train()

    best_score = 1e5 # monitor the loss or the criteria (mse / mae)
    best_epoch = 0
    step = 0

    for epoch in range(args.epochs):
        epoch_loss = []
        loop = tqdm(enumerate(train_dataset), total = len(train_dataset), leave=False)
        for batch_idx, (image, annotations_maps, count) in loop:
            image, count_gt = image.to(args.device), count.to(args.device)
            # zero the parameter gradients - this method was reported as a faster way to zero grad
            for param in model.parameters():
                param.grad = None
            mask_out_list, count_out = model(image)
            loss = 0
            for i, mask_out in enumerate(mask_out_list):
                loss += focal_DRN(mask_out, annotations_maps[i])
            loss += count_loss_func(count_out, count_gt)
            epoch_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            step += 1

        scheduler.step(np.mean(epoch_loss))

        if epoch % args.val_interval == 0:
            model.eval()
            val_loss = []
            val_loop = tqdm(enumerate(val_dataset), total=len(val_dataset), leave=False)
            for val_batch_idx, (inputs, labels) in val_loop:
                with torch.no_grad():
                    inputs, labels = inputs.to(args.device), labels.to(args.device)
                    outputs = model(inputs)
                    loss = focal_DRN(outputs, labels)
                    val_loss.append(loss.item())

            mean_val_loss = np.mean(val_loss)

            print(
                'Val. Epoch: {}/{}. loss: {:1.5f}.'.format(
                    epoch, args.epochs, mean_val_loss))

            if mean_val_loss < best_score:
                best_score = mean_val_loss
                best_epoch = epoch
                save_model_path = os.path.join(args.save_checkpoint_path, f'best_{args.model_type}_model.pth')
                torch.save(model.state_dict(), save_model_path)

            model.train()

        # Early stopping
        if epoch - best_epoch > args.es_patience > 0:
            print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(best_epoch, best_score))
            break

    save_model_path = os.path.join(args.save_checkpoint_path, f'final_{args.model_type}_model.pth')
    torch.save(model.state_dict(), save_model_path)



