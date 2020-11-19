# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:17:07 2020

@author: a-kojima
"""

import torch
import torch.nn as nn
from warprnnt_pytorch import RNNTLoss

# ==============================================
# utils
# ==============================================
def frame_stacking(x):
    newlen = len(x) // 3
    stacked_x = x[0:newlen * 3].reshape(newlen, LOGMEL_DIM * 3)
    return stacked_x

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(-0.1, 0.1)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)
    if classname.find('LSTM') != -1:
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param.data)
            if 'bias' in name:
                param.data.fill_(0)
    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)

# ==============================================
# RNNT model
# ==============================================
            
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.uni_lstm = nn.LSTM(input_size=LOGMEL_DIM*3, hidden_size=NUM_HIDDEN_NODES, num_layers=NUM_ENC_LAYERS, batch_first=True, dropout=DROP_OUT, bidirectional=False)
    def forward(self, x, lengths):
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        h, (hy, cy) = self.uni_lstm(x)
        return h


class Prediction(nn.Module):
    def __init__(self):
        super(Prediction, self).__init__()
        self.L_sy = nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES, bias=False)
        self.L_yy = nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES)
        self.L_ys = nn.Embedding(NUM_CLASSES + 1, NUM_HIDDEN_NODES * 4)
        self.L_ss = nn.Linear(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES * 4, bias=False)
        
    def _lstmcell(self, x, c):
        ingate, forgetgate, cellgate, outgate = x.chunk(4, 1)
        c = (torch.sigmoid(forgetgate) * c) + (torch.sigmoid(ingate) * torch.tanh(cellgate))
        s = torch.sigmoid(outgate) * torch.tanh(c)
        return s, c    
                
    def forward(self, target):
        num_labels = target.size(1)
        s = torch.zeros((BATCH_SIZE, NUM_HIDDEN_NODES), device=DEVICE, requires_grad=False)
        c = torch.zeros((BATCH_SIZE, NUM_HIDDEN_NODES), device=DEVICE, requires_grad=False)
        prediction = torch.zeros((BATCH_SIZE, num_labels, NUM_HIDDEN_NODES), device=DEVICE, requires_grad=False)

        for step in range(num_labels):
            y = self.L_yy(torch.tanh(self.L_sy(s)))
            rec_input = self.L_ys(target[:, step]) + self.L_ss(s)
            s, c = self._lstmcell(rec_input, c)
            prediction[:, step] = y
        return prediction

 
class Joint(nn.Module):
    def __init__(self):
        super(Joint, self).__init__()
        self.output = nn.Linear(NUM_HIDDEN_NODES * 2, NUM_CLASSES + 1)
        
    def forward(self, enc_output, pred_output):
        # enc_output: B * T * HIDDEN_DIM
        # pred_output: B * chara_seq * HIDDEN_DIM
        enc_output, lengths = nn.utils.rnn.pad_packed_sequence(enc_output, batch_first=True)
#        print(enc_output.size(), pred_output.size())
        enc_output = enc_output.unsqueeze(dim=2)
        pred_output = pred_output.unsqueeze(dim=1)
        # calc maximum size
        sz = [max(i, j) for i, j in zip(enc_output.size()[:-1], pred_output.size()[:-1])]
        # padding
        enc_output = enc_output.expand(torch.Size(sz+[enc_output.shape[-1]]))
        pred_output = pred_output.expand(torch.Size(sz+[pred_output.shape[-1]]))   
        # concat 
        out = torch.cat((enc_output, pred_output), dim=-1)
        # feed output layer
        out = self.output(out)
        return out        

class RNN_T(nn.Module):
    def __init__(self):
        super(RNN_T, self).__init__()
        self.encoder = Encoder()
        self.prediction = Prediction()
        self.joint = Joint()
    
    def forward(self, speech, lengths, target):
        h_enc = self.encoder(speech, lengths)
        h_pre = self.prediction(target)
        acts = self.joint(h_enc, h_pre)
        return acts            

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    
    # ============================
    # params
    # ============================
    LOGMEL_DIM = 40
    NUM_HIDDEN_NODES = 512
    NUM_ENC_LAYERS = 4
    NUM_DEC_LAYERS = 1
    NUM_CLASSES = 3672 
            
    # ==================================================================
    # training stategy
    # ==================================================================    
    BATCH_SIZE = 15
    MAX_FRAME = 1000 
    MIN_FRAME = 3
    MAX_LABEL = 80    
    DROP_OUT = 0.2
    FIRST_LR = 0
    TARGET_LR = 0.001
    FIX_STEP = 25000 
    SOS = 0
    EOS = 1
    BLANK = 3672 
    DIFF_LR = (TARGET_LR - FIRST_LR) / FIX_STEP
    learning_rate = FIRST_LR
    
    # ==================================================================
    # model
    # ==================================================================    
    model = RNN_T()
    model.apply(init_weight)
    model.train().to(DEVICE)
    
    # ==================================================================
    # loss
    # ==================================================================    
    rnnt_loss = RNNTLoss(blank=BLANK)
    