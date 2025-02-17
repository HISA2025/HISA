import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from enum import Enum

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_dim, num_layers):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads, 
            dim_feedforward=ff_dim
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer_encoder(x)

class Decoder(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Linear(user_embed_dim, poi_embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed):
        x = self.decoder(user_embed)
        x = self.leaky_relu(x)
        return x

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5
 
        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None, mask=None):
        orig_q_size = q.size()
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  
        v = v.transpose(1, 2)  
        k = k.transpose(1, 2).transpose(2, 3)  

        q = q * self.scale
        x = torch.matmul(q, k)  
        if attn_bias is not None:
            x = x + attn_bias
        if mask is not None:
            mask = mask.unsqueeze(1)
            x = x.masked_fill(mask, 0)

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  

        x = x.transpose(1, 2).contiguous()  
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x
    

class EncoderLayer_notime(nn.Module):
    def __init__(
        self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads
    ):
        super(EncoderLayer_notime, self).__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads
        )
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.ffn_norm1 = nn.LayerNorm(hidden_size)
        self.ffn_norm2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None, mask=None):
      
        y = self.self_attention(x, x, x, attn_bias, mask=mask)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm1(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        x = self.ffn_norm2(x)
        return x

import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn
import numpy as np

class RNN(Enum):
    GRU = 0
    RNN = 1
    LSTM = 2
    
    @staticmethod
    def from_string(name):
        if name == 'gru':
            return RNN.GRU
        if name == 'rnn':
            return RNN.RNN
        if name == 'lstm':
            return RNN.LSTM
        raise ValueError('{} not supported'.format(name))
        

class RNNFactory():
    
    def __init__(self, rnn_type_str):
        self.rnn_type = RNN.from_string(rnn_type_str)
    
    def is_lstm(self):
        return self.rnn_type in [RNN.LSTM]
    
        
    def greeter(self):
        if self.rnn_type == RNN.GRU:
            return 'Use pytorch GRU implementation.'
        if self.rnn_type == RNN.RNN:
            return 'Use vanilla RNN implementation.'
        if self.rnn_type == RNN.LSTM:
            return 'Use pytorch LSTM implementation.'
        
    def create(self, hidden_size):
        if self.rnn_type == RNN.GRU:
            return nn.GRU(hidden_size, hidden_size)
        if self.rnn_type == RNN.RNN:
            return nn.RNN(hidden_size, hidden_size)
        if self.rnn_type == RNN.LSTM:
            return nn.LSTM(hidden_size, hidden_size)
        
class S_Module(nn.Module):

    def __init__(self,beta):
        super(S_Module, self).__init__()
        if(beta==0):
            self.lambda_s = nn.Parameter(torch.rand(1))
        else:
            self.lambda_s = nn.Parameter(torch.ones(1)*beta)

    def forward(self, delta_s):
        return torch.exp(-(delta_s*self.lambda_s))



class T_Module(nn.Module):

    def __init__(self,alpha):
        super(T_Module, self).__init__()
        if(alpha==0):
            self.lambda_t = nn.Parameter(torch.rand(1))
        else:
            self.lambda_t = nn.Parameter(torch.ones(1)*alpha) 

    def forward(self, delta_t):
        return ((torch.cos(delta_t*2*np.pi/86400) + 1) / 2)*torch.exp(-(delta_t/86400*self.lambda_t))#86400 表示一天24小时对应的总秒数





class FlashbackPlusPlus(nn.Module):
    ''' GRU based rnn. applies weighted average using spatial and temporal data WITH user embeddings'''
    
    def __init__(self, input_size, cate_input_size,type_input_size,user_count, hidden_size, f_t, f_s, RNNFactory,setting):
        super().__init__()
        self.input_size = input_size
        self.cate_input_size = cate_input_size
        self.type_input_size = type_input_size
        self.user_count = user_count
        self.hidden_size = hidden_size
        self.f_t = f_t # function for computing temporal weight
        self.f_s = f_s # function for computing spatial weight

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.cate_encoder = nn.Embedding(cate_input_size, hidden_size)
        self.type_encoder = nn.Embedding(type_input_size, hidden_size)
        self.user_encoder = nn.Embedding(user_count, hidden_size)
        self.rnn = RNNFactory.create(hidden_size)
        self.fc = nn.Linear(2*hidden_size, input_size) # create outputs in lenght of locations
        self.setting = setting

        self.seq_model = EncoderLayer_notime(
                                setting.hidden_dim,
                                setting.transformer_nhid,
                                setting.transformer_dropout,
                                setting.attention_dropout_rate,
                                setting.transformer_nhead)
        self.fc = nn.Linear(2 * hidden_size, input_size)
        self.cate_fc = nn.Linear(2*hidden_size, cate_input_size) # create outputs in lenght of category
        self.type_fc = nn.Linear(2*hidden_size, type_input_size) # 

        self.decoder = nn.Linear(setting.hidden_dim, setting.hidden_dim)
        
       
      
        #self.poi_cate_cos_weight = nn.Parameter(torch.rand(1))  
        

    # def Loss_l2(self):
    #     base_params = dict(self.named_parameters())
    #     loss_l2=0.
    #     count=0
    #     for key, value in base_params.items():
    #         if 'bias' not in key and 'pre_model' not in key:
    #             loss_l2+=torch.sum(value**2)
    #             count+=value.nelement()
    #     return loss_l2
        

    def forward(self, x, c,type,t, s, y_t, y_s, h, c_h,type_h,active_user):        

        seq_len, user_len = x.size()
        x_emb = self.encoder(x)    
        c_emb = self.encoder(c)   
        type_emb = self.encoder(type)   
     
        out, h = self.rnn(x_emb, h)
        c_out, c_h = self.rnn(c_emb, c_h)
        type_out, type_h = self.rnn(type_emb, type_h)
     
        out_notime = self.seq_model(x_emb).to(x.device)
        c_out_notime = self.seq_model(c_emb).to(x.device)
        type_out_notime = self.seq_model(type_emb).to(x.device)
     
        out_notime = self.decoder(out_notime.transpose(0,1)).to(x.device).transpose(0,1) 
        c_out_notime = self.decoder(c_out_notime.transpose(0,1)).to(x.device).transpose(0,1) 
        type_out_notime = self.decoder(type_out_notime.transpose(0,1)).to(x.device).transpose(0,1) 
        out_w = torch.zeros(seq_len, user_len, self.hidden_size, device=x.device)
        c_out_w = torch.zeros(seq_len, user_len, self.hidden_size, device=x.device)
        type_out_w = torch.zeros(seq_len, user_len, self.hidden_size, device=x.device)
       
        for i in range(seq_len):#i/1
            sum_w = torch.zeros(user_len, 1, device=x.device)
            c_sum_w = torch.zeros(user_len, 1, device=x.device)
            type_sum_w = torch.zeros(user_len, 1, device=x.device)
            for j in range(i+1):
                dist_t = t[i] - t[j]
                dist_s = torch.norm(s[i] - s[j], dim=-1)
                a_j = self.f_t(dist_t)
                b_j = self.f_s(dist_s)
                a_j = a_j.unsqueeze(1)
                b_j = b_j.unsqueeze(1)
                w_j = a_j*b_j + 1e-10 
               
                c_w_j = a_j+1e-10
                type_w_j = b_j+1e-10
                sum_w += w_j
                c_sum_w += c_w_j
                type_sum_w += w_j
            
                out_w[i] += w_j*out_notime[j] 
                c_out_w[i] += c_w_j*c_out_notime[j]
                type_out_w[i] += type_w_j*type_out_notime[j]
              
            # normliaze according to weights
            out_w[i] /= sum_w
            c_out_w[i] /= c_sum_w
            type_out_w[i] /= type_sum_w
           
    
        p_u = self.user_encoder(active_user)
        p_u = p_u.view(user_len, self.hidden_size)
        out_pu = torch.zeros(seq_len, user_len, 2*self.hidden_size, device=x.device)
        c_out_pu = torch.zeros(seq_len, user_len, 2*self.hidden_size, device=x.device)
        type_out_pu = torch.zeros(seq_len, user_len, 2*self.hidden_size, device=x.device)
    

        for i in range(seq_len): #

            out_pu[i] = torch.cat([out_w[i],p_u], dim=1)
            c_out_pu[i] = torch.cat([c_out_w[i], p_u], dim=1) 
            type_out_pu[i] = torch.cat([type_out_w[i], p_u], dim=1) 

        cosine_similarity = F.linear(F.normalize(out_pu), F.normalize(self.fc.weight))
        cate_cosine_similarity = F.linear(F.normalize(c_out_pu), F.normalize(self.cate_fc.weight))
        type_cosine_similarity = F.linear(F.normalize(type_out_pu), F.normalize(self.type_fc.weight))#
       
        y_linear = self.fc(out_pu)#
        c_y_linear = self.cate_fc(c_out_pu)
        type_y_linear = self.type_fc(type_out_pu)

        return y_linear, c_y_linear,type_y_linear,h,c_h,type_h,cosine_similarity,cate_cosine_similarity,type_cosine_similarity
       