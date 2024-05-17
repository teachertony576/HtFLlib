import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clienttgp_pal import clientTGP_PAL
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
from collections import defaultdict
from torch.utils.data import DataLoader
from utils.data_utils import get_random_batch

class FedTGP_PAL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientTGP_PAL)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes

        self.server_learning_rate = args.local_learning_rate
        self.batch_size = args.batch_size
        self.server_epochs = args.server_epochs
        self.margin_threthold = args.margin_threthold

        self.feature_dim = args.feature_dim
        self.server_hidden_dim = self.feature_dim
        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            PROTO = Trainable_prototypes(
                self.num_classes, 
                self.server_hidden_dim, 
                self.feature_dim, 
                self.device
            ).to(self.device)
            save_item(PROTO, self.role, 'PROTO', self.save_folder_name)
            print(PROTO)
        self.CEloss = nn.CrossEntropyLoss()
        self.MSEloss = nn.MSELoss()

        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        self.min_gap = None
        self.max_gap = None
        #PAL部分
        self.weight = torch.ones(size=(self.num_clients, self.num_clients)) / self.num_clients#初始化个性标签影响因子
        self.public_data=self.load_public_data()#返回的DataLoader类型
        self.temperature = args.temperature


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
            
            #PAL部分
            clients_logits=self.receive_logits()
            self.update_c(clients_logits)
            self.weight_trans(self.weight, clients_logits)
            for client in self.selected_clients:
                client.train_publicdata()

            self.receive_protos()
            self.update_Gen()

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
        

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        uploaded_protos_per_client = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            protos = load_item(client.role, 'protos', client.save_folder_name)
            for k in protos.keys():
                self.uploaded_protos.append((protos[k], k))
            uploaded_protos_per_client.append(protos)

        # calculate class-wise minimum distance
        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        avg_protos = proto_cluster(uploaded_protos_per_client)
        for k1 in avg_protos.keys():
            for k2 in avg_protos.keys():
                if k1 > k2:
                    dis = torch.norm(avg_protos[k1] - avg_protos[k2], p=2)
                    self.gap[k1] = torch.min(self.gap[k1], dis)
                    self.gap[k2] = torch.min(self.gap[k2], dis)
        self.min_gap = torch.min(self.gap)
        for i in range(len(self.gap)):
            if self.gap[i] > torch.tensor(1e8, device=self.device):
                self.gap[i] = self.min_gap
        self.max_gap = torch.max(self.gap)
        print('class-wise minimum distance', self.gap)
        print('min_gap', self.min_gap)
        print('max_gap', self.max_gap)
            
    def update_Gen(self):
        PROTO = load_item(self.role, 'PROTO', self.save_folder_name)
        Gen_opt = torch.optim.SGD(PROTO.parameters(), lr=self.server_learning_rate)
        PROTO.train()
        for e in range(self.server_epochs):
            proto_loader = DataLoader(self.uploaded_protos, self.batch_size, 
                                      drop_last=False, shuffle=True)
            for proto, y in proto_loader:
                y = torch.Tensor(y).type(torch.int64).to(self.device)

                proto_gen = PROTO(list(range(self.num_classes)))

                features_square = torch.sum(torch.pow(proto, 2), 1, keepdim=True)
                centers_square = torch.sum(torch.pow(proto_gen, 2), 1, keepdim=True)
                features_into_centers = torch.matmul(proto, proto_gen.T)
                dist = features_square - 2 * features_into_centers + centers_square.T
                dist = torch.sqrt(dist)
                
                one_hot = F.one_hot(y, self.num_classes).to(self.device)
                gap2 = min(self.max_gap.item(), self.margin_threthold)
                dist = dist + one_hot * gap2
                loss = self.CEloss(-dist, y)

                Gen_opt.zero_grad()
                loss.backward()
                Gen_opt.step()

        print(f'Server loss: {loss.item()}')
        self.uploaded_protos = []
        save_item(PROTO, self.role, 'PROTO', self.save_folder_name)

        PROTO.eval()
        global_protos = defaultdict(list)
        for class_id in range(self.num_classes):
            global_protos[class_id] = PROTO(torch.tensor(class_id, device=self.device)).detach()
        save_item(global_protos, self.role, 'global_protos', self.save_folder_name)


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
            new_logits_i=0.7*new_logits_i+0.3*clients_logits[i]
            save_item(new_logits_i, self.clients[i].role, 'pal_logits', self.clients[i].save_folder_name)#存个性化标签






def proto_cluster(protos_list):
    proto_clusters = defaultdict(list)
    for protos in protos_list:
        for k in protos.keys():
            proto_clusters[k].append(protos[k])

    for k in proto_clusters.keys():
        protos = torch.stack(proto_clusters[k])
        proto_clusters[k] = torch.mean(protos, dim=0).detach()

    return proto_clusters
            

class Trainable_prototypes(nn.Module):
    def __init__(self, num_classes, server_hidden_dim, feature_dim, device):
        super().__init__()

        self.device = device

        self.embedings = nn.Embedding(num_classes, feature_dim)
        layers = [nn.Sequential(
            nn.Linear(feature_dim, server_hidden_dim), 
            nn.ReLU()
        )]
        self.middle = nn.Sequential(*layers)
        self.fc = nn.Linear(server_hidden_dim, feature_dim)

    def forward(self, class_id):
        class_id = torch.tensor(class_id, device=self.device)

        emb = self.embedings(class_id)
        mid = self.middle(emb)
        out = self.fc(mid)

        return out