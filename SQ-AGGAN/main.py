#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import random
import time
import datetime
import dateutil.tz
import torch
import easydict
 
if __name__ == "__main__":
    
    opt = easydict.EasyDict({
        "gpu_id": '0',
        "num_workers":4,
        "epochs":100,
        'batch_size' :4,
        'generator_lr':1e-4,
        'discriminator_lr':1e-4, 
        'total_fold_num':10,
        'fold_num':range(1,11),
    
        "num_sequence":3,
        "image_size":64,
        "feature_dim" :12,

        'nodule_lambda':100.0, 
        'background_lambda': 50.0,
        'feature_lambda': 0.1,
        'yn_lambda': 1.0, 
        
        "data_dir":'MICCAI/MICCAI2021/lung_nodule_sequence/',
        "exp_name":'test1'})

    print(opt)
    

    train_dataset = []
    test_dataset =[]
    for fold in range(opt.total_fold_num):
        train_dataset.append(NoduleDataset(opt.data_dir,'train', fold, opt.total_fold_num))
        test_dataset.append(NoduleDataset(opt.data_dir,'test', fold, opt.total_fold_num))

    for fold in opt.fold_num:
        opt.exp_name = str(fold) + 'th_fold_' + opt.exp_name

        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M')
        output_dir = './experiments/%s_%s' % (timestamp, opt.exp_name)


        print(fold, '-th fold ::: training start')
        train_dataloader = torch.utils.data.DataLoader(train_dataset[fold], batch_size=opt.batch_size, drop_last=True, shuffle=True, num_workers=opt.num_workers)
        test_dataloader = torch.utils.data.DataLoader(test_dataset[fold], batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
        
        from torch.utils.tensorboard import SummaryWriter
        writer_path = 'log/SequentialSynthetic_%s_%s' % (timestamp, opt.exp_name)
        os.makedirs(writer_path)
        writer = SummaryWriter(writer_path)
        
        #hist = model.fit(X_train, Y_train, epochs=1000, batch_size=10, validation_data=(X_val, Y_val))

        algo = sequentialSynthesisTrainer(writer, opt.epochs, opt.gpu_id, opt.batch_size, opt.discriminator_lr, opt.generator_lr, opt.num_sequence, opt.image_size, opt.feature_dim, opt.nodule_lambda, opt.background_lambda, opt.feature_lambda, opt.yn_lambda, output_dir, train_dataloader, test_dataloader)
    
        start_t = time.time()
        algo.train()
        end_t = time.time()

        print(fold, '-th fold ::: total time for training: ', end_t - start_t)

