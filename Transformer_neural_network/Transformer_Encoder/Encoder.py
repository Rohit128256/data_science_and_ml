import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Defining the scale product function to get the final value of each attention head of a token
def scaled_dot_product(q, k, v, mask = None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled,dim=-1)
    new_values = torch.matmul(attention,v)
    return new_values , attention


# Defining the multihead attention class
class MultiheadAttention(nn.Module):
    def __init__(self,dim_model,n_heads):
        super().__init__()
        self.d_model = dim_model
        self.num_head = n_heads
        self.d_head  = dim_model//n_heads
        self.attention_layer = nn.Linear(dim_model,3*dim_model)
        self.linear_layer = nn.Linear(dim_model,dim_model)
    def forward(self, x , mask = None):
        batch_size, seq_len , dim_model = x.size()
        qkv = self.attention_layer(x)
        qkv = qkv.reshape(batch_size,seq_len,self.num_head,3*self.d_head)
        qkv = qkv.permute(0,2,1,3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values.permute(0,2,1,3)
        values = values.reshape(batch_size,seq_len,self.num_head*self.d_head)
        out = self.linear_layer(values)
        return out
    

# After the multihead attention output we have to normalize it for the stable training
class layernormalization(nn.Module):
    def __init__(self, parameters_shape, eps = 1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))
    
    def forward(self,inputs):
        dims = [-(i+1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims,keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdims=True)
        std = (var + self.eps).sqrt()
        y = (inputs-mean)/std
        out = self.gamma * y + self.beta
        return out 
    

#This is a normal Feed Forward neural network
class feedforward(nn.Module):
    def __init__(self, dim_model, hidden, drop_prob = 0.15):
        super().__init__()
        self.linear1 = nn.Linear(dim_model,hidden)
        self.linear2 = nn.Linear(hidden,dim_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    

# Defining the Single Encoder-Layer class
class EncoderLayer(nn.Module):
    def __init__(self,dim_model,ffn_hidden,num_heads, drop_prob):
        super().__init__()
        self.attention = MultiheadAttention(dim_model,n_heads=num_heads)
        self.norm1 = layernormalization(parameters_shape=[dim_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = feedforward(dim_model=dim_model,hidden=ffn_hidden,drop_prob=drop_prob)
        self.norm2 = layernormalization(parameters_shape=[dim_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        orig_x = x
        x = self.attention(x,mask=None)
        x = self.dropout1(x)
        x = self.norm1(x+orig_x)
        orig_x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x+orig_x)
        return x
    

# Defining the Encoder class
class Encoder(nn.Module):
    def __init__(self,dim_model,ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(dim_model,ffn_hidden,num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self,x):
        x = self.layers(x)
        return x
    

# Defining the various attributes
d_model = 96
fnn_hidden = 200
n_heads = 8
drop_prob = 0.12
num_layers = 8


# transformer model object
trans_ecn = Encoder(dim_model=d_model,ffn_hidden=fnn_hidden,num_heads=n_heads,drop_prob=drop_prob,num_layers=num_layers)


# Printing the number of parameters of above model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Print the number of parameters
num_params = count_parameters(trans_ecn)
print(f'Total number of parameters: {num_params}')