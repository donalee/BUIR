import torch
import torch.nn as nn
import torch.nn.functional as F

class BUIR_ID(nn.Module):
    def __init__(self, user_count, item_count, latent_size, momentum):
        super(BUIR_ID, self).__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.latent_size = latent_size
        self.momentum = momentum
        
        self.user_online = nn.Embedding(self.user_count, latent_size)
        self.user_target = nn.Embedding(self.user_count, latent_size)
        self.item_online = nn.Embedding(self.item_count, latent_size)
        self.item_target = nn.Embedding(self.item_count, latent_size)

        self.predictor = nn.Linear(latent_size, latent_size)
   
        self._init_model()
        self._init_target()
    
    def _init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.normal_(m.bias.data)

            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight.data)

    def _init_target(self):        
        for param_o, param_t in zip(self.user_online.parameters(), self.user_target.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False
        
        for param_o, param_t in zip(self.item_online.parameters(), self.item_target.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False
        
    def _update_target(self):
        for param_o, param_t in zip(self.user_online.parameters(), self.user_target.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)
        
        for param_o, param_t in zip(self.item_online.parameters(), self.item_target.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)

    def forward(self, inputs):
        user, item = inputs['user'], inputs['item']

        u_online = self.predictor(self.user_online(user))
        u_target = self.user_target(user)
        i_online = self.predictor(self.item_online(item))
        i_target = self.item_target(item)
        
        return u_online, u_target, i_online, i_target
    
    @torch.no_grad()
    def get_embedding(self):
        u_online = self.user_online.weight
        i_online = self.item_online.weight
        return self.predictor(u_online), u_online, self.predictor(i_online), i_online

    def get_loss(self, output):
        u_online, u_target, i_online, i_target = output

        u_online = F.normalize(u_online, dim=-1)
        u_target = F.normalize(u_target, dim=-1)
        i_online = F.normalize(i_online, dim=-1)
        i_target = F.normalize(i_target, dim=-1)
        
        # Euclidean distance between normalized vectors can be replaced with their negative inner product
        loss_ui = 2 - 2 * (u_online * i_target).sum(dim=-1)
        loss_iu = 2 - 2 * (i_online * u_target).sum(dim=-1)

        return (loss_ui + loss_iu).mean()

