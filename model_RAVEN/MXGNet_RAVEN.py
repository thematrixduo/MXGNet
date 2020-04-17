import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .residual import ResConv,ResConvReason,ResConvInfer,BasicBlock
from .MXEdge import MXEdge


class Model(nn.Module):
    def __init__(self,A,B,  device='cuda',num_fl_1d=5,batch_size = 64,beta=10):
        super(Model,self).__init__()
        self.A = A
        self.B = B
        self.batch_size = batch_size
        self.num_fl_1d = num_fl_1d
        self.num_fl = num_fl_1d * num_fl_1d
        self.device = device
        self.beta = beta
        self.rel_size = 64
        self.enc_size = 32
        self.g_size = 24
        self.tag_tensor = None

        self.encoder_conv = ResConv(BasicBlock,[1,1],[32,self.enc_size])

        self.conv_node = nn.Sequential(
              nn.Conv2d(self.enc_size,self.enc_size,kernel_size=4,stride=2,padding=1),
              nn.BatchNorm2d(self.enc_size),
              nn.ReLU(True),

              nn.Conv2d(self.enc_size,self.enc_size,kernel_size=4,stride=2,padding=1),
              nn.BatchNorm2d(self.enc_size),
              nn.ReLU(True)
              )

        self.moe_layer = MXEdge(in_dim=(self.enc_size+self.num_fl)*2,out_dim=self.g_size,T=self.num_fl)

        self.upsample = nn.Sequential(
              nn.ConvTranspose2d(self.g_size,self.g_size,kernel_size=4,stride=2,padding=1),
              nn.BatchNorm2d(self.g_size),
              nn.ReLU(True)
              )

        self.relation_conv = ResConvReason(BasicBlock,[2,1],[128,self.rel_size],in_dim=self.enc_size*3,g_dim = self.g_size)
        self.infer_conv = ResConvInfer(BasicBlock,[2],[128],256,in_dim=self.rel_size*10)

        self.dropout = nn.Dropout(0.5)
        self.infer_fc = nn.Linear(256,8)
        self.meta_conv = nn.Conv2d(self.rel_size,9,kernel_size=5)
        self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

    def create_tag(self):
        idx = torch.arange(0,self.num_fl).expand(self.batch_size*16,self.num_fl)
        idx = idx.contiguous().unsqueeze(2)
        tag_tensor = torch.zeros(self.batch_size*16,self.num_fl,self.num_fl).scatter_(2,idx,1).float().to(self.device)        
        return tag_tensor


    def encoder_net(self,x):
        conv_out = self.encoder_conv(x)
        conv_out_p = self.conv_node(conv_out).view(-1,self.enc_size,self.num_fl).permute(0,2,1)
        conv_out_p = torch.cat([conv_out_p,self.tag_tensor],-1)
        conv_out_rep = conv_out_p.unsqueeze(1)
        conv_out_rep = conv_out_rep.repeat(1,self.num_fl,1,1)
        
        return conv_out,conv_out_rep




    def relation_infer(self,fg0,fg1,fg2,fl0,fl1,fl2):        
        fg_cat = torch.cat([fg0.squeeze(),fg1.squeeze(),fg2.squeeze()],1)
        fl0 = fl0.squeeze()
        fl1 = fl1.squeeze()
        fl2 = fl2.squeeze() 

        fl_02 = torch.cat([fl0,fl2.permute(0,2,1,3)],-1).view(-1,self.enc_size*2+self.num_fl*2)
        fl_12 = torch.cat([fl1,fl2.permute(0,2,1,3)],-1).view(-1,self.enc_size*2+self.num_fl*2)
        fl_sum = self.moe_layer(fl_02,fl_12)
        fl_sum = fl_sum.view(-1,self.num_fl_1d,self.num_fl_1d,self.g_size).permute(0,3,1,2).contiguous()
        fl_sum = self.upsample(fl_sum)
        f_rel = self.relation_conv(fg_cat,fl_sum)

        return f_rel

    def forward_actual(self,x):
        self.batch_size = x.size(0)

        if self.tag_tensor is None:
            self.tag_tensor = self.create_tag()

        f_g,f_l = self.encoder_net(x.view(-1,1,self.A,self.B))
        f_g_list = torch.split(f_g.view(-1,16,self.enc_size,f_g.size(2),f_g.size(3)),1,1)
        f_l_list = torch.split(f_l.view(-1,16,self.num_fl,self.num_fl,self.enc_size+self.num_fl),1,1)

        r_h_1 = self.relation_infer(f_g_list[0],f_g_list[1],f_g_list[2],f_l_list[0],f_l_list[1],f_l_list[2])
        r_h_2= self.relation_infer(f_g_list[3],f_g_list[4],f_g_list[5],f_l_list[3],f_l_list[4],f_l_list[5])
        score_list = []
        r_h_list =[r_h_1,r_h_2]
     
  
        r_sum = r_h_1 + r_h_2 

        meta_pred = F.sigmoid(self.meta_conv(r_sum))

        for i in range(8):
            r_h_3 = self.relation_infer(f_g_list[6],f_g_list[7],f_g_list[8+i],f_l_list[6],f_l_list[7],f_l_list[8+i]) 
            r_h_list.append(r_h_3)
 
        h_cb = self.infer_conv(torch.cat(r_h_list,1)).squeeze()         
        h_score = self.infer_fc(self.dropout(h_cb))

        return F.log_softmax(h_score),meta_pred.squeeze()

    def forward(self,x,label,meta_target):
        score_vec,meta_pred = self.forward_actual(x)
        criterion = nn.NLLLoss()
        meta_crit = nn.BCELoss()
        aux_loss = meta_crit(meta_pred,meta_target)
        loss = criterion(score_vec,label) + self.beta * aux_loss
        return loss,score_vec
