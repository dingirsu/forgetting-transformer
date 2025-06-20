
import pickle
import math
import torch
from einops import rearrange
from forgetting_transformer.ops.forgetting_attention import forgetting_attention
from torch.nn.functional import cosine_similarity

def precision_metric(quant_o, fa2_o, verbose=True, round_num=4): 
    x, xx = quant_o.float(), fa2_o.float() 
    sim = cosine_similarity(x.reshape(1, -1), xx.reshape(1, -1)).item()
    l1 =   ( (x - xx).abs().sum() / xx.abs().sum() ).item()
    rmse = torch.sqrt(torch.mean((x -xx) ** 2)).item()
    sim = round(sim, round_num)
    l1 = round(l1, round_num)
    rmse = round(rmse, round_num)
    if verbose: print(f'Cossim: {sim:.6f}, L1: {l1:.6f}, RMSE:{rmse:.6f}')
    return {"Cossim": sim, "L1": l1, "RMSE": rmse}

def simu_fox_attn(q, k, v, gate):
    log_lambda = torch.cumsum(gate, dim=-1, dtype=log_fgate.dtype).float()
    S = q @ k.transpose(-2, -1) + log_lambda
    mask = torch.triu(torch.ones(S.size(-2), S.size(-1), device=S.device), diagonal=1).bool()
    S = S.masked_fill(mask, float('-inf'))
    P = S.softmax(dim = -1)
    return P @ v

id = 12
with open(f"tensor/{id}_q.pkl", "rb") as f:
    q = pickle.load(f).cuda()
with open(f"tensor/{id}_k.pkl", "rb") as f:
    k = pickle.load(f).cuda()
with open(f"tensor/{id}_v.pkl", "rb") as f:
    v = pickle.load(f).cuda()
with open(f"tensor/{id}_gk.pkl", "rb") as f:
    log_fgate = pickle.load(f).cuda()    


o = forgetting_attention(
                q, k, v,
                log_fgate,
                head_first=True,
                sm_scale=1 / math.sqrt(q.shape[-1]),
            )
o_simu = simu_fox_attn(q, k, v, log_fgate)
print(precision_metric(o, o_simu))