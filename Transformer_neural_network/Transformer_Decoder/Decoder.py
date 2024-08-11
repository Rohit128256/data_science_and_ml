import torch
import math
from torch import nn
import torch.nn.functional as F

def scaled_dot_product(q,k,v,mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q,k.transpose(-1,-2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled,dim=-1)
    values = torch.matmul(attention,v)
    return values, attention

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob):
        super(FeedForward,self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden,d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class LayerNormalization(nn.Module):
    def __init__(self,parameter_shape,eps=1e-5):
        super().__init__()
        self.parameters_shape = parameter_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameter_shape))
        self.beta = nn.Parameter(torch.zeros(parameter_shape))

    def forward(self,inputs): #shape_inputs:(batch,seqlen,dim_model)
        dims = [-(i+1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims,keepdims=True) #shape_mean:(batch,seqlen,1)
        var = ((inputs - mean)**2).mean(dim=dims,keepdims=True) #shape_mean:(batch,seqlen,1)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma*y + self.beta
        return out
    

class MultiheadcrossAttention(nn.Module):

    def __init__(self,d_model,num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model,2*d_model)
        self.q_layer = nn.Linear(d_model,d_model)
        self.linear_layer = nn.Linear(d_model,d_model)

    def forward(self,x,y,mask=None):
        batch_size, sequence_length, d_model = x.size()
        kv = self.kv_layer(x)
        q = self.q_layer(y)
        kv = kv.reshape(batch_size,sequence_length,self.num_heads,2*self.head_dim)
        q = q.reshape(batch_size,sequence_length,self.num_heads,self.head_dim)
        kv = kv.permute(0,2,1,3)
        q = q.permute(0,2,1,3)
        k, v = kv.chunk(2,dim=-1)
        values, attention = scaled_dot_product(q,k,v,mask) #shape_values:(batch,heads,seqlen,head_dim)
        values = values.permute(0,2,1,3) #shape_values:(batch,seqlen,num_heads,head_dim)
        values = values.reshape(batch_size,sequence_length,self.num_heads*self.head_dim)
        out = self.linear_layer(values)
        return out
    

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model,3*d_model)
        self.linear_layer = nn.Linear(d_model,d_model)

    def forward(self, x, mask=None):
        batch_size, sequence_length, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size,sequence_length,self.num_heads,3*self.head_dim)
        qkv = qkv.permute(0,2,1,3)
        q, k, v = qkv.chunk(3,dim=-1)
        values, attention = scaled_dot_product(q,k,v,mask) #shape_values:(batch,heads,seqlen,head_dim)
        values = values.permute(0,2,1,3) #shape_values:(batch,seqlen,num_heads,head_dim)
        values = values.reshape(batch_size,sequence_length,self.num_heads*self.head_dim)
        out = self.linear_layer(values)
        return out
    


class DecoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,num_heads,drop_prob):
        super().__init__()
        self.self_attention = MultiheadAttention(d_model=d_model,num_heads=num_heads)
        self.norm1 = LayerNormalization(parameter_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.cross_attention = MultiheadcrossAttention(d_model=d_model,num_heads=num_heads)
        self.norm2 = LayerNormalization(parameter_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = FeedForward(d_model=d_model,hidden=ffn_hidden,drop_prob=drop_prob)
        self.norm3 = LayerNormalization(parameter_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self,x,y,decoder_mask):
        y_org = y
        y = self.self_attention(y,decoder_mask)
        y = self.dropout1(y)
        y = self.norm1(y+y_org)

        y_org = y
        y = self.cross_attention(x,y,mask=None)
        y = self.dropout2(y)
        y = self.norm2(y+y_org)

        y_org = y
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.norm2(y+y_org)
        return y
    

class SequentialDecoder(nn.Sequential):
    def forward(self,*inputs):
        x, y, mask = inputs
        for module in self._modules.values():
            y = module(x,y,mask)
        return y
    
class Decoder(nn.Module):
    def __init__(self,d_model,ffn_hidden,num_heads,drop_prob,num_layers=1):
        super().__init__()
        self.layers = SequentialDecoder(*[DecoderLayer(d_model,ffn_hidden,num_heads,drop_prob) for _ in range(num_layers)])
        
    def forward(self, x, y, mask):
        y = self.layers(x,y,mask)
        return y
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


dim_model = 512
num_heads = 8
drop_prob = 0.1
batch_size = 32
max_sequence_length = 200
ffn_hidden = 2048
num_layers = 5

x = torch.randn((batch_size,max_sequence_length,dim_model))
y = torch.randn((batch_size,max_sequence_length,dim_model))
mask = torch.full([max_sequence_length,max_sequence_length],float('-inf'))
mask = torch.triu(mask,diagonal=1)
decoder = Decoder(d_model=dim_model,ffn_hidden=ffn_hidden,num_heads=num_heads,drop_prob=drop_prob,num_layers=num_layers)

# Print the number of parameters
num_params = count_parameters(decoder)
print(f'Total number of parameters: {num_params}')