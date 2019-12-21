import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

from loss.tools import *
from config.config_coral import *
from models.resnet_coral import *
from dataset.dataset_coral import Dataset
torch.backends.cudnn.deterministic = True


def train_model(model, train_loader, opt, optimizer):
    '''
    '''
    start_time = time.time()
    model.train()
    for batch_idx, (features, targets, levels) in enumerate(train_loader):
        features = features.to(opt.device)
        targets = targets
        targets = targets.to(opt.device)
        levels = levels.to(opt.device)
            
        # FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = cost_fn(logits, levels, imp)
        optimizer.zero_grad()
        cost.backward()
        # UPDATE MODEL PARAMETERS
        optimizer.step()
        # LOGGING
        if not batch_idx % 50:
            s = ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                % (epoch+1, opt.max_epoch, batch_idx,
                len(train_dataset)//opt.batch_size, cost))
            print(s)
    s = 'Time elapsed: %.2f min' % ((time.time() - start_time)/60)
    print(s)

def test_model(model, test_loader, DEVICE):
    '''
    '''
    start_time = time.time()
    model.eval()
    with torch.set_grad_enabled(False):  # save memory during inference

        #train_mae, train_mse = compute_mae_and_mse(model, train_loader,
        #                                       device=DEVICE)
        test_mae, test_mse = compute_mae_and_mse(model, test_loader,
                                             device=DEVICE)

        s = 'MAE/RMSE: | Test: %.2f/%.2f' % (test_mae, torch.sqrt(test_mse))
        print(s)

    s = 'Total Training Time: %.2f min' % ((time.time() - start_time)/60)
    print(s)

if __name__ == '__main__':
    opt = Config()
    #imp = task_importance_weights('dataset/deepcam_coral_train_list.txt', opt.num_classes)
    imp = torch.ones(opt.num_classes-1, dtype=torch.float)
    imp = imp.to(opt.device)

    train_dataset = Dataset(root=opt.root, 
                            data_list_file=opt.train_list, 
                            num_classes=opt.num_classes, 
                            phase='train', 
                            input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=opt.num_workers)
    
    test_dataset = Dataset(root=opt.root, 
                           data_list_file=opt.test_list, 
                           num_classes=opt.num_classes, 
                           phase='test', 
                           input_shape=opt.input_shape)
    testloader = data.DataLoader(test_dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=opt.num_workers)

    model = resnet18(opt.num_classes, False)
    model = model.to(opt.device)

    if opt.resume == True:
        model.load_state_dict(torch.load(opt.resume_model))
    
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    scheduler = MultiStepLR(optimizer, milestones = opt.milestones, gamma=0.1)

    for epoch in range(opt.max_epoch):
        scheduler.step(epoch)
        train_model(model, trainloader, opt, optimizer)
        test_model(model, testloader, opt.device)
        
        ########## SAVE MODEL #############
        #model = model.to(torch.device('cpu'))
        torch.save(model.state_dict(), os.path.join('checkpoints/', 'coral_age_epoch_' + str(epoch) + '.pth'))

