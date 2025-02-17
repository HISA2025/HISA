import torch
from torch import nn
import numpy as np
from utils import *
from network_category_type_CL_transformer import FlashbackPlusPlus
from scipy.sparse import csr_matrix
from network_category_type_CL_transformer import S_Module, T_Module, FlashbackPlusPlus
import torch.nn.functional as F


class FlashbackPlusPlusTrainer():
    
    def __init__(self, lambda_t, lambda_s,alpha_start=1.0, alpha_end=0.1, total_epochs=100,decay_type='linear',weight_type='user_weight'):
        
        self.lambda_t = lambda_t
        self.lambda_s = lambda_s
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.decay_type = decay_type  
        self.total_epochs = total_epochs
        self.weight_type = weight_type
        
        
        self.alpha_initial = 1.0

        self.alpha_max = 10.0      
        self.beta_initial = 0.5  
        self.beta_min = 0.0     

    def parameters(self):
       
        params = list(self.model.parameters())\
            + list(self.t_module.parameters())\
            + list(self.s_module.parameters())
        return params
        
    
    def prepare(self, loc_count, individual_loc_count,cate_count,type_count,user_count, hidden_size, RNNFactory, device,setting):
        self.s_module = S_Module(self.lambda_s).to(device)
        self.t_module = T_Module(self.lambda_t).to(device)
        f_t = lambda delta_t: self.t_module(delta_t)
        f_s = lambda delta_s: self.s_module(delta_s)
            
        
        self.loc_count = loc_count
        self.cate_count = cate_count
        self.type_count = type_count
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.each_sample_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.model = FlashbackPlusPlus(loc_count,cate_count,type_count,user_count, hidden_size, f_t, f_s, RNNFactory,setting).to(device)

    
    def evaluate(self, x,c, type,t, s, y_t, y_s, h,c_h,type_h, active_users):
      
        self.model.eval()
        out, c_out,type_out, h,c_h,type_h, cosine,cate_cosine,type_cosine= self.model(x, c, type,t, s, y_t, y_s, h,c_h,type_h, active_users)
        out_t = out.transpose(0, 1)
        c_out_t = c_out.transpose(0, 1)
        type_out_t = type_out.transpose(0, 1)
        return out_t, c_out_t,type_out_t,h,c_h,type_h 
    
    
    def loss(self, x,c,type, t, s, y,c_y,type_y, y_t, y_s, h,c_h,type_h, active_users):
        ''' takes a batch (users x location sequence)
        and corresponding targets in order to compute the training loss '''
     
        self.model.train()
        out, c_out,type_out,h , c_h , type_h,poi_cosine_similarity,cate_cosine_similarity,type_cosine_similarity= self.model(x, c,type, t, s, y_t, y_s, h, c_h,type_h, active_users)
       
        out_pre=out.transpose(0, 1)
        c_out_pre=c_out.transpose(0, 1)
        type_out_pre=type_out.transpose(0, 1)
       
        out = out.view(-1, self.loc_count)
        c_out = c_out.view(-1, self.cate_count)
        type_out = type_out.view(-1, self.type_count)

        y = y.view(-1)
        c_y = c_y.view(-1)
        type_y = type_y.view(-1)
        ###############################cosine similarity#############
        #######################for poi ################
       
        poi_cosine_similarity = poi_cosine_similarity.view(-1, self.loc_count)
        type_cosine_similarity = type_cosine_similarity.view(-1, self.type_count)
 
        target_cosine = poi_cosine_similarity.gather(1, y.unsqueeze(1)).view(-1)
        type_target_cosine = type_cosine_similarity.gather(1, type_y.unsqueeze(1)).view(-1)
      
 
        #####################for category ##################
    
        cate_cosine_similarity = cate_cosine_similarity.view(-1, self.cate_count)
        cate_target_cosine = cate_cosine_similarity.gather(1, c_y.unsqueeze(1)).view(-1)

        l = self.cross_entropy_loss(out, y)
        c_l = self.cross_entropy_loss(c_out, c_y)
        type_l = self.cross_entropy_loss(type_out, type_y)

        poi_difficulty = 2 * (target_cosine - target_cosine.min().item()) / (target_cosine.max().item() - target_cosine.min().item()) - 1
        cate_difficulty = 2 * (cate_target_cosine - cate_target_cosine.min().item()) / (cate_target_cosine.max().item() - cate_target_cosine.min().item()) - 1
        type_difficulty = 2 * (type_target_cosine - type_target_cosine.min().item()) / (type_target_cosine.max().item() - type_target_cosine.min().item()) - 1
          

        return l,c_l,type_l,poi_difficulty,cate_difficulty,type_difficulty,h,c_h,type_h,out_pre,c_out_pre,type_out_pre
