import torch

class Config(object):
    optimizer = 'sgd'                 # optimizer should be sgd, adam
    num_workers = 4                   # how many workers for loading data
    print_freq = 50                   # print info every N batch
    batch_size = 256                  # batch size
    milestones = [20, 100, 150]       # adjust lr 

    lr = 0.001                        # initial learning rate
    warmup = False                    # warm up
    max_epoch = 200                   # max epoch
    num_classes = 100
    device = torch.device('cuda:5')
    input_shape = (3, 112, 112)

    root = 'dataset/'
    test_list  = 'dataset/deepcam_coral_test_list.txt'
    train_list = 'dataset/deepcam_coral_train_list.txt'
    
    resume = True
    resume_model = 'checkpoints/coral_age_epoch_199.pth'
