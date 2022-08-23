import os
import numpy as np
import torch
import wandb
from tqdm import tqdm
from counters.MSR_DRN.utils.losses import focal_DRN, calc_metrices

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
        epoch_loss = 0
        loop = tqdm(enumerate(train_dataset), total = len(train_dataset), leave=False)
        for batch_idx, (image, annotations_maps, count) in loop:
            image, count_gt = image.to(args.device), count.to(args.device)
            # zero the parameter gradients - this method was reported as a faster way to zero grad
            for param in model.parameters():
                param.grad = None
            mask_out_list, count_out = model(image)
            loss = 0
            for i, mask_out in enumerate(mask_out_list):
                loss += focal_DRN(mask_out[0,0], annotations_maps[0,i].to(args.device))
            loss += count_loss_func(count_out, count_gt)
            epoch_loss += loss.item()
            loop.set_postfix({'mean loss': epoch_loss/(batch_idx+1)})
            loss.backward()
            optimizer.step()
            step += 1

        scheduler.step(epoch_loss/len(train_dataset))

        if epoch % args.val_interval == 0:
            model.eval()
            gt_pred_pairs = {
                'gt':[],
                'pred':[]
            }
            val_loop = tqdm(enumerate(val_dataset), total=len(val_dataset), leave=False)
            for batch_idx, (image, annotations_maps, count_gt) in val_loop:
                with torch.no_grad():
                    image, count = image.to(args.device), count_gt.to(args.device)
                    val_mask_out_list, val_count_out = model(image)
                    gt_pred_pairs['gt'].append(count)
                    gt_pred_pairs['pred'].append(val_count_out)
                    val_loss = 0
                    for i, mask_out in enumerate(val_mask_out_list):
                        val_loss += focal_DRN(mask_out[0,0], annotations_maps[0,i].to(args.device))
                    val_loss += count_loss_func(val_count_out, count)
            mean_val_loss = val_loss/len(val_dataset)

            print(
                'Val. Epoch: {}/{}. loss: {:1.5f}.'.format(
                    epoch, args.epochs, mean_val_loss))
            metric = calc_metrices(gt_pred_pairs, args.monitor_metric)

            if metric < best_score:
                best_score = metric
                best_epoch = epoch
                save_model_path = os.path.join(args.save_trained_models, f'{args.model_type}_{args.exp_number}', f'best_{args.model_type}_model.pth')
                torch.save(model.state_dict(), save_model_path)
            model.train()

        # Early stopping
        if epoch - best_epoch > args.es_patience > 0:
            print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(best_epoch, best_score))
            break

    # save_model_path = os.path.join(args.save_checkpoint_path, f'final_{args.model_type}_model.pth')
    # torch.save(model.state_dict(), save_model_path)



