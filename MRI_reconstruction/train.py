import os
import pickle
import torch
from torch import nn
from model import UnetModel
from data_preprocess import *
from torch.utils.data import TensorDataset, DataLoader
from fastmri import losses as losses
import time
import cv2

from fastmri.losses import SSIMLoss
from skimage.measure import compare_psnr
from skimage.metrics import structural_similarity

from skimage.measure import compare_ssim
from torch.nn import functional as F


if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

#print(device)
file_dir = "E:\singlecoil_val"





# load the data
training_data, train_gt, validation_data, validation_gt, testing_data, testing_gt = load_data_final(file_dir_path=file_dir)
train_dataset = TensorDataset(training_data, train_gt)
validation_dataset = TensorDataset(validation_data, validation_gt)
testing_dataset = TensorDataset(testing_data, testing_gt)
if __name__ == '__main__':
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=8,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=0)
    '''''''''
    validation_loader = DataLoader(dataset=validation_dataset,
                              batch_size=32,
                              shuffle=True,
                              num_workers=2)
    testing_loader = DataLoader(dataset=testing_dataset,
                              batch_size=32,
                              shuffle=True,
                              num_workers=2)

    '''

    model = UnetModel(
        in_chans=1,
        out_chans=1,
        chans=32,  # 640   # 32
        num_pool_layers=4,
        drop_prob=0
    ).to(device)
    lr = 0.001
    lr_step_size = 40
    lr_gamma = 0.1
    weight_decay = 0.0
    epoches = 1501
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)


    def lr_scheduler(optimizer, epoch):
        """decay learning rate by a factor of 0.5 every 5000"""
        if epoch % 1400 == 0 and epoch > 55:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            print('LR is set to {}'.format(param_group['lr']))


    def mse(gt, pred):
        """ Compute Mean Squared Error (MSE) """
        return np.mean((gt - pred) ** 2)


    def nmse(gt, pred):
        """ Compute Normalized Mean Squared Error (NMSE) """
        return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


    def psnr(gt, pred):
        """ Compute Peak Signal to Noise Ratio metric (PSNR) """
        return compare_psnr(gt, pred, data_range=gt.max())


    def ssim(gt, pred):
        """ Compute Structural Similarity Index Metric (SSIM). """
        return structural_similarity(
            gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
        )


    '''''''''
    def train(epoch):
        model.train()
        train_total_loss = 0
        train_total_ssim = 0
        for iteration, samples in enumerate(train_loader):
            print('iteration {} out of {} in training'.format(iteration, epoch))
            train_x, train_y = samples
            train_x = train_x.unsqueeze(1).to(device)
            train_y = train_y.unsqueeze(1).to(device)
            output = model(train_x)
            loss = F.l1_loss(output, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total_loss = train_total_loss + loss.item()
            ssim_value = SSIMLoss(train_y.squeeze().cpu().numpy(), output.squeeze().cpu().numpy())
            train_total_ssim = train_total_ssim + ssim_value
            print(train_total_loss)
            print(train_total_ssim)

    '''

    train_av_epoch_loss_list = []
    train_av_epoch_ssim_list = []
    loss_fun = nn.L1Loss().to(device)
    # loss_fun = nn.L1Loss(reduction="mean").to(device)

    # training process
    '''''''''
    model_path = "E:\model\sense_recon_1000.pth"
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # start_epoch = checkpoint['epoch']
    '''
    print('succssfully load the model')
    model.train()
    for epoch in range(epoches):
        print('Epoch {}'.format(epoch))
        start = time.time()
        # model.train()
        train_total_loss = 0
        train_total_ssim = 0
        i = 0
        for iteration, samples in enumerate(train_loader):
            # print('iteration {} out of {} in training'.format(iteration, epoch))
            train_y, train_x = samples
            '''''''''
            train_x = torch.squeeze(train_x)
            train_y = torch.squeeze(train_y)

            print(train_x.size())
            print(train_y.size())
            train_x = train_x.numpy()
            train_y = train_y.numpy()
            plt.figure()
            plt.imshow(train_x, cmap='gray')               # to show the x and y image
            plt.figure()
            plt.imshow(train_y,cmap='gray')
            plt.show()

            train_x = torch.tensor(train_x)
            train_y = torch.tensor(train_y)
            '''
            train_x = train_x.unsqueeze(1).to(device)
            train_y = train_y.unsqueeze(1).to(device)
            #print(train_x)
            #print(torch.max(train_x))
            # print(train_x.shape)
            #  print(train_y.shape)
            output = model(train_x)
            # print(output.shape)
            '''''''''
            output = torch.squeeze(output)
            output = output.detach().cpu().numpy()
            plt.figure()
            plt.imshow(output, cmap='gray')
            plt.show()
            '''
            # B = output.squeeze().data.cpu().numpy()
            # print(B)
            # B = B[0:1,:,:]
            # C = train_y.squeeze().squeeze().squeeze(0).cpu().numpy()[0:1,:,:]
            train_y = train_y.squeeze(1)
            output = output.squeeze(1)
            #print(output.shape)
            loss = loss_fun(train_y, output)
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total_loss = train_total_loss + loss.item()
            #  ssim_value = ssim(train_y, train_x)
            # print(ssim_value)
            # train_total_ssim = train_total_ssim + ssim_value
            #  print(train_total_loss)
            # print(train_total_ssim)
            i = iteration
        end = time.time()
        print(str(end - start) + 'seconds')
        train_av_loss = train_total_loss / (i + 1)
        print('train_av_loss is '+str(train_av_loss))
        # train_av_ssim = train_total_ssim /(i + 1)
        train_av_epoch_loss_list.append(train_av_loss)
        # train_av_epoch_ssim_list.append(train_total_ssim)

        lr_scheduler(optimizer, epoch)

        if epoch % 50 == 0 and epoch > 0:
            print('save the model at epoch {}'.format(epoch))
            model_dir = file_dir
            if not (os.path.exists(model_dir)): os.makedirs(model_dir)
            torch.save(model.state_dict(), "{0}/sense_recon_{1:03d}.pth".format(model_dir, epoch))
        #    print("epoch: ", epoch, "train_av_epoch_ssim_list: ", train_av_epoch_ssim_list)
            print("epoch: ", epoch, "train_av_epoch_loss_list: ", train_av_epoch_loss_list)







