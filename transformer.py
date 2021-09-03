import torch
import torch.nn as nn
import math
from torch.autograd import Variable


# Transformer Model

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=128, kernel_size=14, stride=3, padding=2)
        self.batch1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=14, stride=3, padding=0)
        self.batch2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=10, stride=2, padding=0)
        self.batch3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=10, stride=2, padding=0)
        self.batch4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=10, stride=1, padding=0)
        self.batch5 = nn.BatchNorm1d(256)
        self.conv6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=10, stride=1, padding=0)
        self.batch6 = nn.BatchNorm1d(256)
        self.relu = torch.nn.ReLU()
    def forward(self,x):
      x = self.relu(self.batch1(self.conv1(x)))
      x = self.relu(self.batch2(self.conv2(x)))
      x = self.relu(self.batch3(self.conv3(x)))
      x = self.relu(self.batch4(self.conv4(x)))
      x = self.relu(self.batch5(self.conv5(x)))
      x = self.relu(self.batch6(self.conv6(x)))
      x = x.permute(0,2,1)
      return x

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 500):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len],requires_grad=False)
        return x

class Transformer(nn.Module):
    def __init__(self,d_model, n_head, dim_feedforward, n_layers, n_class, dropout_trans, dropout_fc):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
        self.n_layer = n_layers
        self.n_class = n_class
        self.dropout_trans = dropout_trans
        self.dropout_fc = dropout_fc
        self.embedd = Embedding()
        self.pe = PositionalEncoder(256)
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, dropout=dropout_trans,batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layers,num_layers=n_layers)
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.Dropout(dropout_fc),
                                       nn.Linear(128, n_class))
    def forward(self, x):
      x = self.embedd(x)
      x = self.pe(x)
      x = self.encoder(x)
      x = torch.mean(x,dim=1)
      x = self.fc(x)
      return x