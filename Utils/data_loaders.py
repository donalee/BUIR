import torch
import torch.utils.data as data

class ImplicitFeedback(data.Dataset):
    def __init__(self, user_count, item_count, interaction_mat):
        super(ImplicitFeedback, self).__init__()
        
        self.user_count = user_count
        self.item_count = item_count
    
        self.interactions = []
        for user in interaction_mat:
            for item in interaction_mat[user]:
                self.interactions.append([user, item, 1])
 
    def __len__(self):
        return len(self.interactions)
        
    def __getitem__(self, idx):
        return {
            'user': self.interactions[idx][0], 
            'item': self.interactions[idx][1], 
        }

