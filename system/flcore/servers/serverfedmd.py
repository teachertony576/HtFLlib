
import time
import numpy as np
from flcore.clients.clientfedmd import clientfedmd
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
from collections import defaultdict
from utils.data_utils import get_random_batch
import torch.nn.functional as F
import torch


class FedMD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.temperature = args.temperature
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientfedmd)
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
            self.weight_trans(clients_logits)
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

        uploaded_logits = []
        for client in self.selected_clients:
            logits = load_item(client.role, 'logits', client.save_folder_name)
            uploaded_logits.append(logits)
        return uploaded_logits#这是个列表list
    

    def weight_trans(self, clients_logits):
        clients_logits=sum(clients_logits) / len(clients_logits)
        save_item(clients_logits, self.role, 'global_logits', self.save_folder_name)#存全局标签






