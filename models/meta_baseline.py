import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import models
import utils
from .models import register

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


@register('meta-baseline')
class MetaBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={}, CrossAtt_args={}, method='cos',
                 temp=10., temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method
              
        hdim = 1600
        self.v_att = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)       
               
        self.n = 55
        act_fn = nn.LeakyReLU
                
        ##############
        ## AttFEX Module ##
        # 1x1 Convs representing M(.), N(.)
        self.fe = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(  # 64 --> args.wm
                self.n, 1), stride=(1, 1), padding='valid', bias=False),
            act_fn(),
                    
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(  # 64 --> args.wm, 32 --> args.wn
                1, 1), stride=(1, 1), padding='valid', bias=False),
            act_fn())
            
            #nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(  # 64 --> args.wm, 32 --> args.wn
                #1, 1), stride=(1, 1), padding='valid', bias=False),
            #act_fn()
            #)
            
        # Query, Key and Value extractors as 1x1 Convs
        self.f_q = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(  # 32 --> args.wn
            1, 1), stride=(1, 1), padding='valid', bias=False)
        self.f_k = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(  # 32 --> args.wn
            1, 1), stride=(1, 1), padding='valid', bias=False)
        self.f_v = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(  # 32 --> args.wn
            1, 1), stride=(1, 1), padding='valid', bias=False)
        ##############
        
       
        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp
        
    def forward(self, x_shot, x_query):
              
        #0.2 h_att-v_att-avg version-2
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        bs = x_shot.shape[0]
        n_ways = x_shot.shape[1]
        k_shots = x_shot.shape[2]
        q_shots = x_query.shape[1]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_shot_len = len(x_shot)
        x_query_len = len(x_query)
        
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))   
        
        #pre-v_att
        x_tot_shape = x_tot.shape[-3:]        
        x_tot_vatt = x_tot.flatten(start_dim=1)
        x_shot_vatt = x_tot_vatt[:x_shot_len]   
        x_query_vatt = x_tot_vatt[-x_query_len:]   
        
        x_query_before = x_query_vatt.squeeze(dim=0)
        x_query_num = x_query_before.cpu().numpy()
        np.savetxt("query_before.txt",x_query_num)
        
        x_shot_vatt = x_shot_vatt.view(bs, x_shot_len, x_shot_vatt.shape[-1])   #1,25,1600
        x_query_vatt = x_query_vatt.view(bs, x_query_len, x_query_vatt.shape[-1])   #1,50,1600
        	                                     
        #v_att        
        x_shot_vatt = self.v_att(x_shot_vatt, x_shot_vatt, x_shot_vatt)
        x_query_vatt = self.v_att(x_query_vatt, x_query_vatt, x_query_vatt)
        x_tot_vatt = torch.cat([x_shot_vatt, x_query_vatt], dim=1)
        x_tot_vatt = x_tot_vatt.view(bs*(n_ways*k_shots+q_shots), *x_tot_shape)
        
        #h_att
        x_shot_hatt = x_tot[:x_shot_len]   
        x_shot_hatt = x_shot_hatt.view(bs, n_ways, k_shots, *x_tot_shape)
        x_shot_hatt = x_shot_hatt.mean(dim=2)
        x_shot_hatt = x_shot_hatt.view(bs*n_ways, *x_tot_shape)
        x_query_hatt = x_tot[-x_query_len:]
        x_tot_hatt = torch.cat([x_shot_hatt, x_query_hatt], dim=0)
        w_h = x_tot_hatt.permute(2, 3, 0, 1)
        w_h = w_h.reshape(w_h.shape[0] * w_h.shape[1],
                      w_h.shape[2], w_h.shape[3]).unsqueeze(dim=1)
        w_h = self.fe(w_h)
        
        xq = self.f_q(w_h)
        xk = self.f_k(w_h)
        xv = self.f_v(w_h)                          
        
        xq = xq.squeeze(dim=1).squeeze(dim=1).transpose(
            0, 1).reshape(-1, x_tot_hatt.shape[2], x_tot_hatt.shape[3])
        xk = xk.squeeze(dim=1).squeeze(dim=1).transpose(
            0, 1).reshape(-1, x_tot_hatt.shape[2], x_tot_hatt.shape[3])
        xv = xv.squeeze(dim=1).squeeze(dim=1).transpose(
            0, 1).reshape(-1, x_tot_hatt.shape[2], x_tot_hatt.shape[3])

        #h_att Attention Block
        xq = xq.reshape(xq.shape[0], xq.shape[1]*xq.shape[2])
        xk = xk.reshape(xk.shape[0], xk.shape[1]*xk.shape[2])
        xv = xv.reshape(xv.shape[0], xv.shape[1]*xv.shape[2])

        w_h = torch.mm(xq, xk.transpose(0, 1)/xk.shape[1]**0.5)
        softmax = nn.Softmax(dim=-1)
        w_h = softmax(w_h)
        w_h = torch.mm(w_h, xv)

        #h_att Transductive Mask transformed input        
        w_h = w_h.reshape(-1, x_tot_hatt.shape[2], x_tot_hatt.shape[3])
        x_tot_hatt_vatt = x_tot_vatt * w_h
        x_tot_hatt_vatt = x_tot_hatt_vatt.flatten(start_dim=1)
        x_tot_hatt_vatt_s = x_tot_hatt_vatt[:x_shot_len]
        x_tot_hatt_vatt_q = x_tot_hatt_vatt[-x_query_len:] 
        
        x_tot_hatt_vatt_s = x_tot_hatt_vatt_s.view(bs, n_ways, k_shots, x_tot_hatt_vatt.shape[-1])
        x_tot_hatt_vatt_q = x_tot_hatt_vatt_q.view(bs, x_query_len, x_tot_hatt_vatt.shape[-1])
        
        #shot and query
        x_shot = x_tot_hatt_vatt_s
        x_query = x_tot_hatt_vatt_q                 
        
        #distance
        if self.method == 'cos':
            x_shot = x_shot.mean(dim=-2)
            x_shot = F.normalize(x_shot, dim=-1)
            x_query = F.normalize(x_query, dim=-1)
            
            x_shot1 = x_shot.squeeze(dim=0)
            x_shot_num = x_shot1.cpu().numpy()
            np.savetxt("shot.txt",x_shot_num)
            
            x_query_after = x_query.squeeze(dim=0)
            x_query_num = x_query_after.cpu().numpy()
            np.savetxt("query_after.txt",x_query_num)
            
            metric = 'dot'
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'
        
        logits = utils.compute_logits(
                x_query, x_shot, metric=metric, temp=self.temp)
        return logits
        
