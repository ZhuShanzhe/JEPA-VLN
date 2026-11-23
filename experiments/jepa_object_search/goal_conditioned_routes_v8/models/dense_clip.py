"""Local RN50 visual features. This module does not accept segmentation or future states."""
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


class DenseObserved:
    def __init__(self,encoder,center,basis,anchors):
        self.encoder=encoder
        self.center=center;self.basis=basis;self.anchors=anchors

    @torch.inference_mode()
    def local(self,images):
        clip=self.encoder.net.visual
        pool=clip.attnpool
        captured=[]
        hook=pool.register_forward_pre_hook(lambda _,args:captured.append(args[0]))
        output=[]
        try:
            for j in range(0,len(images),12):
                x=torch.stack([self.encoder.prep(Image.fromarray(np.ascontiguousarray(arr)))
                               for arr in images[j:j+12]])
                clip(x)
                grid=captured.pop().flatten(2).transpose(1,2)
                if grid.size(1)!=49:raise ValueError("expected final RN50 7x7 feature map")
                tokens=F.normalize(pool.c_proj(pool.v_proj(grid)).float(),dim=-1)
                output.append(tokens)
        finally:
            hook.remove()
        return torch.cat(output)

    def compressed(self,local,text):
        return ((local-self.center)@self.basis,local@self.anchors.T,local@text.T)
