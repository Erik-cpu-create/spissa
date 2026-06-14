import torch
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
    
x = torch.arange(8).float()
print("original:", x)
print("rotate_half:", rotate_half(x))
