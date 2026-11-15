"""Use Torch Hub's verified local DINOv2 checkout for the author's SALAD network."""
import os
from pathlib import Path
import runpy
import torch

native = torch.hub.load
folder = Path(os.environ["TORCH_HOME"])/"hub/facebookresearch_dinov2_main"
assert (folder/"hubconf.py").is_file()

def cached_hub(repo_or_dir, model, *args, **kwargs):
    if repo_or_dir in ("facebookresearch/dinov2", "facebookresearch/dinov2:main"):
        kwargs["source"]="local"
        return native(str(folder), model, *args, **kwargs)
    return native(repo_or_dir, model, *args, **kwargs)

torch.hub.load=cached_hub
runpy.run_module("mapping.server",run_name="__main__")
