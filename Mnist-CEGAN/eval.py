import numpy as np
import sys
import os
import utils
import time
import csv
import argparse
import json
import math
import model
import noise
import data
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

def load_data_set(self, batch_size):
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=batch_size)
    return train_dl,val_dl


class EvalOps(object):

    def __init__(self, gpu_num, batch_size):
        self.device = device
        self.batch_size = batch_size

        tmp = str(time.time())
        self.curtime = tmp[:9]
        self.code_start_time = time.time()

        def evaluate(self, extractor_weight_path, classifier_weight_path):
            _, self.data_loader_test = DataLoader(val_data, batch_size=batch_size)

            print('test data loader len: %d' % (len(self.data_loader_test)))

            self.feature_extractor = Feature_Extractor(pretrained_weight=None, num_classes=self.num_labels)
            self.feature_extractor.load_state_dict(torch.load(extractor_weight_path))
            self.feature_extractor.eval()
            self.feature_extractor.to(self.device)

            self.feature_classifier = Feature_Classifier(pretrained_weight=None, num_classes=self.num_labels)
            self.feature_classifier.load_state_dict(torch.load(classifier_weight_path))
            self.feature_classifier.eval()
            self.feature_classifier.to(self.device)

            self.criterion = nn.CrossEntropyLoss()

            loss = 0.0
            step_acc = 0.0
            total_inputs_len = 0
            with torch.no_grad():  # 이 안에서 행해지는 모든 코드들은 미분 적용이 안 됨
                for idx, (inputs, labels) in enumerate(self.data_loader_test):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    extracted_feature = self.feature_extractor(inputs)
                    logits_real = self.feature_classifier(extracted_feature)
                    _, preds = torch.max(logits_real, 1)
                    loss += self.criterion(logits_real, labels).item() * inputs.size(0)

                    corr_sum = torch.sum(preds == labels.data)
                    step_acc += corr_sum.double()
                    total_inputs_len += inputs.size(0)
                    if idx % 20 == 0:
                        print('%d / %d' % (idx, len(self.data_loader_test)))

            loss /= total_inputs_len
            step_acc /= total_inputs_len

            print('Test loss: [%.6f] accuracy: [%.4f]' % (loss, step_acc))
            print("time: %.3f" % (time.time() - self.code_start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1, help=" 0, 1, 2 or 3 ")

    parser.add_argument("--batch_size", type=int, default=32, help='batch_size')

    # weight path
    parser.add_argument("--extractor_weight_path", type=str, default='./feature_extractor_target_resnet_caltech.pth',
                        help='extractor weight path')
    parser.add_argument("--classifier_weight_path", type=str, default='./feature_classifier_target_resnet_caltech.pth',
                        help='classifier weight path')

    parser.add_argument("--pretrained_base_model", type=str, default='', help='pretrained_base_model')

    # './checkpoint/vgg16_cifar10_source.pth'
    args = parser.parse_args(args=[])

    test = EvalOps(gpu_num=args.gpu, mbatch_size=args.batch_size)

    test.evaluate(args.extractor_weight_path, args.classifier_weight_path)
