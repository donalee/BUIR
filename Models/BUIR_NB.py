import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BUIR_NB(nn.Module):
    def __init__(self, user_count, item_count, latent_size, norm_adj, momentum, n_layers=3, drop_flag=False):
        super(BUIR_NB, self).__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.latent_size = latent_size
        self.momentum = momentum
        
        self.online_encoder = LGCN_Encoder(user_count, item_count, latent_size, norm_adj, n_layers, drop_flag)
        self.target_encoder = LGCN_Encoder(user_count, item_count, latent_size, norm_adj, n_layers, drop_flag)

        self.predictor = nn.Linear(latent_size, latent_size)
        
        self._init_target()
    
    def _init_target(self):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False 
    
    def _update_target(self):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)

    def forward(self, inputs):
        u_online, i_online = self.online_encoder(inputs)
        u_target, i_target = self.target_encoder(inputs) 
        return self.predictor(u_online), u_target, self.predictor(i_online), i_target 

    @torch.no_grad()
    def get_embedding(self):
        u_online, i_online = self.online_encoder.get_embedding()
        return self.predictor(u_online), u_online, self.predictor(i_online), i_online

    def get_loss(self, output):
        u_online, u_target, i_online, i_target = output

        u_online = F.normalize(u_online, dim=-1)
        u_target = F.normalize(u_target, dim=-1)
        i_online = F.normalize(i_online, dim=-1)
        i_target = F.normalize(i_target, dim=-1)
        
        loss_ui = 2 - 2 * (u_online * i_target).sum(dim=-1)
        loss_iu = 2 - 2 * (i_online * u_target).sum(dim=-1)

        return (loss_ui + loss_iu).mean()
    

class LGCN_Encoder(nn.Module):
    def __init__(self, user_count, item_count, latent_size, norm_adj, n_layers=3, drop_flag=False):
        super(LGCN_Encoder, self).__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.latent_size = latent_size
        self.layers = [latent_size] * n_layers 

        self.norm_adj = norm_adj
        
        self.drop_ratio = 0.2
        self.drop_flag = drop_flag

        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.user_count, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.item_count, self.latent_size))),
        })

        return embedding_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).cuda()
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).cuda()
        return out * (1. / (1 - rate))

    def forward(self, inputs):
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    np.random.random() * self.drop_ratio,
                                    self.sparse_norm_adj._nnz()) if self.drop_flag else self.sparse_norm_adj

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
            
        for k in range(len(self.layers)):
            ego_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        user_all_embeddings = all_embeddings[:self.user_count, :]
        item_all_embeddings = all_embeddings[self.user_count:, :]

        users, items = inputs['user'], inputs['item']
        user_embeddings = user_all_embeddings[users, :]
        item_embeddings = item_all_embeddings[items, :]

        return user_embeddings, item_embeddings

    @torch.no_grad()
    def get_embedding(self):
        A_hat = self.sparse_norm_adj
        
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
            
        for k in range(len(self.layers)):
            ego_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings = all_embeddings[:self.user_count, :]
        item_all_embeddings = all_embeddings[self.user_count:, :]

        return user_all_embeddings, item_all_embeddings
