import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
from collections import defaultdict
from utils.data_utils import VanillaKDLoss
import torch.nn.functional as F

class clientTGP_PAL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda
        self.T=args.nontarget_temperature
        self.distill_loss_fn = VanillaKDLoss(temperature=args.temperature)
        self.publicdata_batch_size = args.publicdata_batch_size


    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        prev_model = load_item(self.role, 'prev_model', self.save_folder_name)
        save_item(model, self.role, 'prev_model', self.save_folder_name)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        # model.to(self.device)
        model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        criterionKL = nn.KLDivLoss(reduction='batchmean')
        criterionKL.to(self.device)
        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = model.base(x)
                output = model.head(rep)
                loss = self.loss(output, y)

                #非目标蒸馏
                # #soft_outputs=F.softmax(output / self.T, dim=1)
                # non_targets_mask = torch.ones(self.publicdata_batch_size, self.num_classes).to(self.device).scatter_(1, y.view(-1, 1), 0)
                # non_target_soft_outputs = output[non_targets_mask.bool()].view(self.publicdata_batch_size, self.num_classes - 1)
                # with torch.no_grad():#减少显卡的使用
                #     prev_output = prev_model(x)
                #     #soft_prev_outpus = F.softmax(prev_output / self.T, dim=1)#使用self.distill_loss_fn可以不用softmax操作
                #     non_target_soft_prev_outputs = prev_output[non_targets_mask.bool()].view(self.publicdata_batch_size, self.num_classes  - 1)
                
                # inon_target_loss = self.distill_nontarget_fn(non_target_soft_outputs, non_target_soft_prev_outputs)
                # #inon_target_loss = inon_target_loss * (self.T ** 2)#会造成loss变得很大
                # loss+=inon_target_loss

                if global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_protos[y_c]) != type([]):
                            proto_new[i, :] = global_protos[y_c].data
                    # we use MSE here following FedProto's official implementation, where lamda is set to 10 by default.
                    # see https://github.com/yuetan031/FedProto/blob/main/lib/update.py#L171
                    loss += self.loss_mse(proto_new, rep) * self.lamda
                    
                    #Prototype-Exemplar Distillation/PED loss /原型-示例蒸馏
                    with torch.no_grad():#减少显卡的使用
                        prev_rep = prev_model.base(x)
                        # dot_product = torch.matmul(prev_rep, (rep-proto_new).T)
                        # loss -= 0.1*torch.mean(torch.diagonal(dot_product))
                        kl_div_loss = criterionKL(F.log_softmax(prev_rep-proto_new, dim=1), F.softmax(rep-proto_new, dim=1))
                        loss += kl_div_loss
                        




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

        self.collect_protos()
        save_item(model, self.role, 'model', self.save_folder_name)
        save_item(logits, self.role, 'logits', self.save_folder_name)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def collect_protos(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = model.base(x)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        save_item(agg_func(protos), self.role, 'protos', self.save_folder_name)

    def test_metrics(self):
        testloader = self.load_test_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        model.eval()

        test_acc = 0
        test_num = 0

        if global_protos is not None:
            with torch.no_grad():
                for x, y in testloader:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    rep = model.base(x)

                    output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
                    for i, r in enumerate(rep):
                        for j, pro in global_protos.items():
                            if type(pro) != type([]):
                                output[i, j] = self.loss_mse(r, pro)

                    test_acc += (torch.sum(torch.argmin(output, dim=1) == y)).item()
                    test_num += y.shape[0]

            return test_acc, test_num, 0
        else:
            return 0, 1e-5, 0

    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
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
                rep = model.base(x)
                output = model.head(rep)
                loss = self.loss(output, y)

                if global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_protos[y_c]) != type([]):
                            proto_new[i, :] = global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda
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


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos