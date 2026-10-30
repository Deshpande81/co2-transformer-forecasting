"""Direct six-output prediction; no anchors, residuals, or monotonic penalty."""
import torch
from torch import nn
from .transformer import EncoderBlock,PositionalEncoding
class CO2Forecaster(nn.Module):
    def __init__(self,n_features=54,width=32,layers=2,dropout=.1,mode='combined',target_mean=None,target_scale=None):
        super().__init__()
        if mode not in ('combined','process','co2'):raise ValueError(mode)
        self.mode=mode; d={'combined':54,'process':30,'co2':24}[mode]
        self.proj=nn.Linear(d,width);self.pos=PositionalEncoding(width)
        self.blocks=nn.ModuleList([EncoderBlock(width,4,dropout) for _ in range(layers)])
        self.norm=nn.LayerNorm(width)
        self.head=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Dropout(dropout),nn.Linear(width,6))
        self.positive=nn.Softplus()
        # These are TRAINING-ONLY output calibration constants, not historical CO2 anchors.
        # The network predicts an unrestricted nonlinear mapping from input sequences.
        mean=torch.as_tensor(target_mean if target_mean is not None else [1.]*6,dtype=torch.float32).clamp_min(1e-5)
        scale=torch.as_tensor(target_scale if target_scale is not None else [1.]*6,dtype=torch.float32).clamp_min(1e-4)
        self.register_buffer('output_center',mean+torch.log(-torch.expm1(-mean))) # inverse softplus
        self.register_buffer('output_scale',scale)
        # Zero-init last layer: initial output = train-location mean; NOT a persistence anchor.
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,x):
        if self.mode=='process':x=x[...,:30]
        elif self.mode=='co2':x=x[...,30:]
        z=self.pos(self.proj(x))
        for b in self.blocks:z=b(z)
        return self.positive(self.output_center + self.output_scale*self.head(self.norm(z[:,-1])))
