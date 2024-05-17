
import time
import numpy as np
from flcore.clients.clientpalfl import clientpalfl
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
from collections import defaultdict
from utils.data_utils import get_random_batch
import torch.nn.functional as F
import torch


class PAL_FL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.temperature = args.temperature
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientpalfl)
        self.weight = torch.ones(size=(self.num_clients, self.num_clients)) / self.num_clients#初始化个性标签影响因子
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.public_data=self.load_public_data()#返回的DataLoader类型
        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes
        self.Tau = args.Tau

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            public_data_random = get_random_batch(self.public_data)#从公共数据集中随机返回一个batch的数据

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate()

            for client in self.selected_clients:
                client.public_data=public_data_random#将全局的公共数据集传递给每个client
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            clients_logits=self.receive_logits()
            self.update_c(clients_logits)
            self.weight_trans(self.weight, clients_logits)
            for client in self.selected_clients:
                client.train_publicdata()



            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        

    def receive_logits(self):
        assert (len(self.selected_clients) > 0)

        self.selected_clients_ids = []
        zero_logits=load_item(self.selected_clients[0].role, 'logits', self.selected_clients[0].save_folder_name)[0] * 0.
        uploaded_logits = [zero_logits for _ in range(self.num_clients)]
        for client in self.selected_clients:
            self.selected_clients_ids.append(client.id)
            logits = load_item(client.role, 'logits', client.save_folder_name)
            uploaded_logits[client.id]=logits
        return uploaded_logits#这是个列表list
    
    def update_c(self,clients_logits):
        # weight: shape as [num_clients, num_clients], element means c_i->c_j weight
        # clients' logits matrix before update c
        #raw_logits_matrix = torch.stack(clients_logits, dim=-1)
        #T_raw_logits_matrix=raw_logits_matrix/self.temperature
        #weighted_logits_matrix = self.weight_trans(self.weight, raw_logits_matrix)
        for self_idx in self.selected_clients_ids:
            self_logits_local=clients_logits[self_idx]
            cos_sim=torch.zeros(self.num_clients,self.num_clients)
            for ids in self.selected_clients_ids:
                cos_sim[self_idx][ids]=torch.mean(torch.cosine_similarity(clients_logits[self_idx]/ self.temperature,clients_logits[ids]/ self.temperature))#先计算相似度
            cos_sim[self_idx][self_idx]=0.0#自己不能蒸馏自己

            cos_sim=F.softmax(cos_sim,dim=1)

            for teach_idx in self.selected_clients_ids:
                self.weight[self_idx][teach_idx]=cos_sim[self_idx][teach_idx]

    def weight_trans(self,weight, clients_logits):
        """weight transport the logits_matrix

        Args:
            weight: shape as [num_clients, num_clients], element means c_i->c_j weight
            logits_matrix: [num_data_alignment, num_classes, num_clients],
                
        """
        num_clients = self.num_clients
        assert num_clients == len(clients_logits), \
            f"weight size {num_clients}, logits long: {clients_logits.size(0)}"

        new_logits_list = []
        for i in self.selected_clients_ids:
            new_logits_i = torch.zeros_like(clients_logits[i])
            for j in self.selected_clients_ids:
                if i == j:
                    continue
                new_logits_i += weight[i][j] * clients_logits[j]
            new_logits_i=(1-self.Tau)*new_logits_i+self.Tau*clients_logits[i]
            save_item(new_logits_i, self.clients[i].role, 'pal_logits', self.clients[i].save_folder_name)#存个性化标签






