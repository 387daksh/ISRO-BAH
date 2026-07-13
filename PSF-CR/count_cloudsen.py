from cloudsen12_models import cloudsen12
import torch

model = cloudsen12.load_model_by_name("dtacs4bands")
params = sum(p.numel() for p in model.parameters())
print(f"CloudSen12 params: {params:,}")
