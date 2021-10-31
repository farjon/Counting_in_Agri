import os
import torch
from tqdm import tqdm
import numpy as np
import datetime
from tensorboardX import SummaryWriter

def train_and_eval(args, train_dataset, val_dataset, model, loss_func, optimizer, scheduler):
    device = args.device
    model.to(device)
    model.train()
    writer_path = os.path.join(args.ROOT_DIR, 'Logs', args.data)
    writer = SummaryWriter(writer_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    best_score = 1e5 # monitor the loss or the criteria (mse / mae)
    best_epoch = 0
    step = 0

    for epoch in range(args.epochs):
        epoch_loss = []
        loop = tqdm(enumerate(train_dataset), total = len(train_dataset), leave=False)
        for batch_idx, (inputs, labels) in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients - this method was reported as a faster way to zero grad
            for param in model.parameters():
                param.grad = None
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            epoch_loss.append(loss.item())
            writer.add_scalars('Loss', {'train': loss.item()}, step)

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
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = loss_func(outputs, labels)
                    val_loss.append(loss.item())

            mean_val_loss = np.mean(val_loss)

            print(
                'Val. Epoch: {}/{}. loss: {:1.5f}.'.format(
                    epoch, args.epochs, mean_val_loss))
            writer.add_scalars('Loss', {'val': mean_val_loss}, step)

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

