#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.utils.data as data
import numpy as np
import os
from medpy.io import load
import pandas as pd
import random
import cv2
import random

class NoduleDataset(data.Dataset):
    def __init__(self, data_dir, mode, fold_number = 0, total_fold = 10):
        self.data_dir = data_dir 
        patient_id = np.array(pd.read_csv(self.data_dir + 'patient_id.csv').patient_id)
        ##np.array(~.변수이름)하면 다 풀어헤쳐져서 나옴
        patient_id = list(sorted(patient_id))
        random.seed(200)
        patient_id = random.sample(patient_id, 300)
        
        test_patient_id = patient_id[fold_number::total_fold] ##fold_num부터 끝까지 total_fold를 간격으로 뽑아서 test로 사용할거다
        
        if mode == "train":
            self.target_patient_id = [i for i in patient_id if i not in test_patient_id]
        elif mode == "test":
            self.target_patient_id = test_patient_id
                                                                 #vol_sequence
        self.vol_filenames = self.load_filenames(self.data_dir + 'seq_scan_64', self.target_patient_id)
        self.bg_filenames = self.load_filenames(self.data_dir + 'bg_sequence', self.target_patient_id)
        self.info = self.load_info(self.data_dir + 'seq_info.csv', self.target_patient_id)



       
    def load_info(self, data_dir, target_patient_id): ##patient의 info를 출력
        csv = pd.read_csv(data_dir)
        info = np.array([csv.patient_id,       ### 0
                        csv.nodule_id,         ### 1
                        csv.total_slice,       ### 2
                        csv.diameter,          ### 3
                        csv.subtlety,          ### 4
                        csv.internalStructure, ### 5
                        csv.calcification,     ### 6
                        csv.sphericity,        ### 7
                        csv.margin,            ### 8
                        csv.lobulation,        ### 9
                        csv.spiculation,       ### 10
                        csv.texture,           ### 11
                        csv.malignancy])       ### 12
        info = np.transpose(info)
        
        target_info = []
        for temp_info in info:
            if target_patient_id.count(temp_info[0]):  #target patient id에 id가 있냐?
                target_info.append(temp_info)
        target_info = np.array(target_info)
        return target_info

    def load_filenames(self, data_dir, target_patient_id): #patient의 file을 출력
        target_filenames = [] 

        filenames = os.listdir(data_dir) #모든 파일 리스트를 출력
        for filename in filenames:
            patient_id = filename.split("_")[0]
            if target_patient_id.count(patient_id):
                target_filenames.append(filename)

        return target_filenames #target patient의 아이디가 있는 file들의 이름을 출력
    
    def find_info(self, filename):
        current_seq_filename = filename.split('_')[0] + '_' + filename.split('_')[1]          ### current sequence name
        slice_num = filename.split('_')[-1].split('.')[0].replace("z","")

        for i, data in enumerate(self.info):
            csv_filename = data[0] + '_' + str(data[1])      ### filename in csv
            if csv_filename == current_seq_filename:
                
                ###feature가 여기 잇음
                # feature = np.array([int(slice_num), data[2],data[3]]) 
                feature = np.array([int(slice_num), data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9],data[10],data[11],data[12]])
                return feature         ### total_slice, diameter, .... , malignancy   >>>  12 features
        return 0

    def __len__(self):
        return len(self.vol_filenames)

    def __getitem__(self, index):
        ### mask     (m)
        mask_path = self.data_dir + '/seq_mask_64/' + self.vol_filenames[index]  
        nodule_mask, _ = load(mask_path)                                                      ### (x, y, seq_slices)
        
        nodule_mask = nodule_mask[:,:,1]
        temp_bg_mask = 1.0 - nodule_mask
        ####################################################  ###############################################################################################################################################################################
        # kernel = np.ones((7, 7), np.uint8)
        # bg_mask = cv2.erode(temp_bg_mask, kernel, iterations=1)
        bg_mask = temp_bg_mask.copy()
        ####################################################
        nodule_mask = np.expand_dims(nodule_mask, axis = 0)
        bg_mask = np.expand_dims(bg_mask, axis = 0)
        
        temp_bg_mask = np.expand_dims(temp_bg_mask, axis = 2)
        temp_bg_mask = np.concatenate((temp_bg_mask, temp_bg_mask, temp_bg_mask), 2)
        
        ### input slice sequence, masked input slice sequnce   (x, x')
        vol_sequence_path = self.data_dir + '/seq_scan_64/' + self.vol_filenames[index]
        vol_sequence, _ = load(vol_sequence_path)                                       ### (x, y, seq_slices)
        
        gt_slice = vol_sequence[:,:,1].copy()
        gt_slice = np.expand_dims(gt_slice, axis = 0)                                   ### (channel, x, y)

        masked_vol_sequence = temp_bg_mask * vol_sequence
        masked_vol_sequence = np.expand_dims(masked_vol_sequence, axis=0)               ### (channel, x, y, seq_slices)
        masked_vol_sequence = np.transpose(masked_vol_sequence, (3,0,1,2))              ### (seq_slices, channel, x, y)     
 
        ### background
        bg_index = random.randrange(0, len(self.bg_filenames))
        bg_sequence_path = self.data_dir + '/bg_sequence/' + self.bg_filenames[bg_index]
        bg_sequence, _ = load(bg_sequence_path)
        # ### ### version 1 ######################################################################
        bg_sequence = np.expand_dims(bg_sequence, axis=0)                               ### (channel, x, y, seq_slices)
        bg_sequence = np.transpose(bg_sequence, (3,0,1,2))                              ### (seq_slices, channel, x, y)     
        ### ### version 2 ######################################################################
        # masked_bg_sequence = temp_bg_mask * bg_sequence
        # masked_bg_sequence = np.expand_dims(masked_bg_sequence, axis=0)                 ### (channel, x, y, seq_slices)
        # masked_bg_sequence = np.transpose(masked_bg_sequence, (3,0,1,2))                ### (seq_slices, channel, x, y)
        # bg_sequence = np.expand_dims(bg_sequence, axis=0)                               ### (channel, x, y, seq_slices)
        # bg_sequence = np.transpose(bg_sequence, (3,0,1,2))                              ### (seq_slices, channel, x, y)     
        ########################################################################################

        ### feature information
        feature_sequence = self.find_info(self.vol_filenames[index])
        
        return masked_vol_sequence, bg_sequence, feature_sequence, gt_slice, nodule_mask, bg_mask
        # return masked_vol_sequence, masked_bg_sequence, bg_sequence, feature_sequence, gt_slice, nodule_mask, bg_mask

