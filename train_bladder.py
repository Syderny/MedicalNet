'''
Training code for Bladder datasets classification
Written by Yinrui
'''

from setting import parse_opts 
from datasets.bladder import BladderDataset
from model_classification import generate_model
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
import time
from utils.logger import log
from scipy import ndimage
import os

def train(train_loader, valid_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, sets):
    # settings
    batches_per_epoch = len(train_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    loss_cl = nn.CrossEntropyLoss()

    print("Current setting is:")
    print(sets)
    print("\n\n")     
    if not sets.no_cuda:
        loss_cl = loss_cl.cuda()

        
    model.train()
    train_time_sp = time.time()
    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))
        
        scheduler.step()
        log.info('lr = {}'.format(scheduler.get_lr()))
        
        loss_list = []
        acc_list = []

        for batch_id, batch_data in enumerate(train_loader):
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volumes, label = batch_data

            if not sets.no_cuda: 
                volumes = volumes.cuda()
                label = label.cuda()

            optimizer.zero_grad()
            output = model(volumes)

            # calculating loss
            loss_value = loss_cl(output, label)
            loss = loss_value
            loss.backward()                
            optimizer.step()
            
            loss_list.append(loss.item())
            loss_avg = sum(loss_list) / len(loss_list)


            # calculating acc
            pred = output.argmax(dim=1)
            acc_list.append(torch.eq(pred, label).sum().float().item())
            acc = sum(acc_list) / len(acc_list)

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                    'Train Batch: {}-{} ({}), loss = {:.3f}, loss_avg = {:.3f}, batch_acc = {:.3f}, avg_batch_time = {:.3f}'\
                    .format(epoch, batch_id, batch_id_sp, loss_value.item(), loss_avg, acc, avg_batch_time))
          
            if not sets.ci_test:
                # save model
                if batch_id == 0 and batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                #if batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                    model_save_path = '{}_epoch_{}_batch_{}.pth.tar'.format(save_folder, epoch, batch_id)
                    model_save_dir = os.path.dirname(model_save_path)
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)
                    
                    log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id)) 
                    torch.save({
                                'ecpoch': epoch,
                                'batch_id': batch_id,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict()},
                                model_save_path)

        loss_list = []
        acc_list = []

        with torch.no_grad():
            for batch_id, batch_data in enumerate(valid_loader):
                # getting data batch
                batch_id_sp = epoch * batches_per_epoch
                volumes, label = batch_data

                if not sets.no_cuda: 
                    volumes = volumes.cuda()
                    label = label.cuda()

                output = model(volumes)

                # calculating loss
                loss_value = loss_cl(output, label)
                loss = loss_value
                
                loss_list.append(loss.item())
                loss_avg = sum(loss_list) / len(loss_list)

                # calculating acc
                pred = output.argmax(dim=1)
                acc_list.append(torch.eq(pred, label).sum().float().item())
                acc = sum(acc_list) / len(acc_list)

                avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
                log.info(
                        'Valid Batch: {}-{} ({}), loss = {:.3f}, loss_avg = {:.3f}, batch_acc = {:.3f}, avg_batch_time = {:.3f}'\
                        .format(epoch, batch_id, batch_id_sp, loss_value.item(), loss_avg, acc, avg_batch_time))
                
                            
    print('Finished training')            
    if sets.ci_test:
        exit()


if __name__ == '__main__':
    # settting
    sets = parse_opts()   
    # if sets.ci_test:
    #     sets.n_epochs = 1
    #     sets.no_cuda = False
    #     sets.data_root = r'D:\data\bladder_data'
    #     sets.pretrain_path = ''
    #     sets.num_workers = 0
    #     sets.model_depth = 10
    #     sets.resnet_shortcut = 'A'
    #     sets.input_D = 50
    #     sets.input_H = 160
    #     sets.input_W = 160
       
    
    # getting model
    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets) 
    print (model)
    # optimizer
    if sets.ci_test:
        params = [{'params': parameters, 'lr': sets.learning_rate}]
    else:
        params = [
                { 'params': parameters['base_parameters'], 'lr': sets.learning_rate }, 
                { 'params': parameters['new_parameters'], 'lr': sets.learning_rate*100 }
                ]
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)   
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    # train from resume
    if sets.resume_path:
        if os.path.isfile(sets.resume_path):
            print("=> loading checkpoint '{}'".format(sets.resume_path))
            checkpoint = torch.load(sets.resume_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
              .format(sets.resume_path, checkpoint['epoch']))

    # getting data
    sets.phase = 'train'
    if sets.no_cuda:
        sets.pin_memory = False
    else:
        sets.pin_memory = True    
    dataset = BladderDataset(sets.data_root, sets)
    len_trainset = int(len(dataset) * 0.7)
    len_validset = len(dataset) - len_trainset
    trainset, validset = random_split(dataset, [len_trainset, len_validset])
    train_loader = DataLoader(trainset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=sets.pin_memory)
    valid_loader = DataLoader(validset, batch_size=sets.batch_size)

    # training
    train(train_loader, valid_loader, model, optimizer, scheduler, total_epochs=sets.n_epochs, save_interval=sets.save_intervals, save_folder=sets.save_folder, sets=sets) 
