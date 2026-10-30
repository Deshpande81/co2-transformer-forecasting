"""Self-implemented attention and encoder; no nn.TransformerEncoder."""
import math
import torch
from torch import nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,width=32,heads=4,dropout=.1):
        super().__init__();assert width%heads==0
        self.heads=heads;self.d=width//heads
        self.qkv=nn.Linear(width,3*width);self.proj=nn.Linear(width,width);self.drop=nn.Dropout(dropout)
    def forward(self,x):
        b,t,d=x.shape
        q,k,v=self.qkv(x).reshape(b,t,3,self.heads,self.d).permute(2,0,3,1,4)
        score=(q@k.transpose(-2,-1))/math.sqrt(self.d)
        # Causal mask: no attention to later positions even within input history.
        future=torch.ones(t,t,device=x.device,dtype=torch.bool).triu(1)
        att=self.drop(score.masked_fill(future,float('-inf')).softmax(-1))
        return self.proj((att@v).transpose(1,2).reshape(b,t,d))

class EncoderBlock(nn.Module):
    def __init__(self,width=32,heads=4,dropout=.1):
        super().__init__();self.n1=nn.LayerNorm(width);self.n2=nn.LayerNorm(width)
        self.att=MultiHeadSelfAttention(width,heads,dropout);self.drop=nn.Dropout(dropout)
        self.ff=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Dropout(dropout),nn.Linear(2*width,width))
    def forward(self,x):
        x=x+self.drop(self.att(self.n1(x)))
        return x+self.drop(self.ff(self.n2(x)))

class PositionalEncoding(nn.Module):
    def __init__(self,width): super().__init__();self.width=width
    def forward(self,x):
        t=torch.arange(x.size(1),device=x.device,dtype=x.dtype)[:,None]
        k=torch.arange(0,self.width,2,device=x.device,dtype=x.dtype)
        freq=torch.exp(-math.log(10000)*k/self.width)
        pe=torch.zeros((x.size(1),self.width),device=x.device,dtype=x.dtype)
        pe[:,0::2]=torch.sin(t*freq)
        pe[:,1::2]=torch.cos(t*freq[:pe[:,1::2].shape[1]])
        return x+pe[None]
