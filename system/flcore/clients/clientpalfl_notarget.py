import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
from collections import defaultdict
from utils.data_utils import VanillaKDLoss
import torch.nn.functional as F

class clientpalfl_notarget(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)
        self.distill_nontarget_fn = VanillaKDLoss(temperature=args.nontarget_temperature)
        self.distill_loss_fn = VanillaKDLoss(temperature=args.temperature)
        self.publicdata_batch_size = args.publicdata_batch_size
        self.lamda = args.lamda
        self.T=args.nontarget_temperature


    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        prev_model = load_item(self.role, 'prev_model', self.save_folder_name)
        save_item(model, self.role, 'prev_model', self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        #global_logits = load_item('Server', 'global_logits', self.save_folder_name)
        
        start_time = time.time()

        # model.to(self.device)
        model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        criterionKL = nn.KLDivLoss(reduction='batchmean')
        criterionKL.to(self.device)
        #使用本地数据集进行训练
        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                    
                output = model(x)
                loss = self.loss(output, y)

                #非目标蒸馏
                #soft_outputs=F.softmax(output / self.T, dim=1)
                non_targets_mask = torch.ones(self.batch_size, self.num_classes).to(self.device).scatter_(1, y.view(-1, 1), 0)
                non_target_soft_outputs = output[non_targets_mask.bool()].view(self.batch_size, self.num_classes - 1)
                with torch.no_grad():#减少显卡的使用
                    prev_output = prev_model(x)
                    #soft_prev_outpus = F.softmax(prev_output / self.T, dim=1)
                    non_target_soft_prev_outputs = prev_output[non_targets_mask.bool()].view(self.batch_size, self.num_classes  - 1)
                
                inon_target_loss = self.distill_nontarget_fn(non_target_soft_outputs, non_target_soft_prev_outputs)
                #inon_target_loss = inon_target_loss * (self.T ** 2)#会造成loss变得很大
                loss+=inon_target_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        #输出公共数据集的logits
        model.eval()
        logits =[]
        with torch.no_grad():#减少显卡的使用
            x,y=self.public_data
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            output = model(x)
            logits.append(output)
            logits = torch.cat(logits, dim=0)

        save_item(model, self.role, 'model', self.save_folder_name)
        save_item(logits, self.role, 'logits', self.save_folder_name)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_logits = load_item('Server', 'global_logits', self.save_folder_name)
        # model.to(self.device)
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)
                loss = self.loss(output, y)

                if global_logits != None:
                    logit_new = copy.deepcopy(output.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_logits[y_c]) != type([]):
                            logit_new[i, :] = global_logits[y_c].data
                    loss += self.loss(output, logit_new.softmax(dim=1)) * self.lamda
                    
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
    

    def train_publicdata(self):
        model = load_item(self.role, 'model', self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        pal_logits = load_item(self.role, 'pal_logits', self.save_folder_name)

        # model.to(self.device)
        model.train()

        x,y=self.public_data
        if type(x) == type([]):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        output = model(x)
        loss = self.distill_loss_fn(output, pal_logits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        save_item(model, self.role, 'model', self.save_folder_name)



