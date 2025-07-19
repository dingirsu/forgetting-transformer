
import pickle
import math
import torch
from einops import rearrange
from forgetting_transformer.ops.forgetting_attention import forgetting_attention
from forgetting_transformer.ops.forgetting_attention_sp import forgetting_attention_sp
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

def flash_attn_fp4(q, k, v, gate, device='cuda', BLOCK_M=128, random_quant=True):
    output_buffer = torch.zeros(v.shape, device=device)
    Q_BLOCKS = torch.split(q, BLOCK_M, dim=-2)
    K_BLOCKS = torch.split(k, BLOCK_M, dim=-2)
    V_BLOCKS = torch.split(v, BLOCK_M, dim=-2)
    bs, head, seqlen, headdim = q.shape
    seqlen = q.shape[-2] // BLOCK_M + (1 if q.shape[2] % BLOCK_M != 0 else 0)
    safe_e_list = []
    for j in range(seqlen):
        qi = Q_BLOCKS[j]
        old_o = output_buffer[...,j  * BLOCK_M: min((j+1) * BLOCK_M, q.shape[2]), :]
        old_d = torch.zeros((bs, head, min(BLOCK_M, q.shape[2] - j * BLOCK_M), 1), device=device)
        old_m = torch.full((bs, head, min(BLOCK_M, q.shape[2] - j * BLOCK_M), 1), -torch.inf, device=device)
        k_block_num = k.shape[-2] // BLOCK_M + (1 if q.shape[2] % BLOCK_M != 0 else 0)
        for i in range(k_block_num):
            kj = K_BLOCKS[i]
            vj = V_BLOCKS[i]
            x_qkt = (qi, kj.transpose(2, 3)) * q.shape[-1]**-0.5
            local_m = torch.max(x_qkt, dim=-1, keepdim=True).values
            new_m = torch.maximum(old_m, local_m)
            safe_e = torch.exp(x_qkt - new_m)
            curr_d = torch.sum(safe_e, dim=-1, keepdim=True)
            safe_e_list.append(safe_e)
            new_d = old_d * torch.exp(old_m - new_m) + curr_d
            new_o = old_o * torch.exp(old_m - new_m) + safe_e @ vj
            old_m = new_m
            old_d = new_d
            old_o = new_o
        output_buffer[...,j  * BLOCK_M: min((j+1) * BLOCK_M, q.shape[2]), :] = old_o / old_d
    return output_buffer

def simu_fox_attn(q, k, v, gate, block_size = 4096):
    log_lambda = torch.cumsum(gate, dim=-1, dtype=log_fgate.dtype).float()
    # log_lambda_blocks = torch.split(log_lambda, block_size, dim = -1)
    # for i in range(len(log_lambda_blocks)):
    
    # threshold = torch.quantile(log_lambda, q=0.01)
    # print(threshold)
    # breakpoint()
    D = log_lambda.unsqueeze(-1) - log_lambda.unsqueeze(-2)
    mask = torch.triu(torch.ones(D.shape[-1], D.shape[-2], device=D.device), diagonal=-block_size)
    D = D.masked_fill(mask == 0, -1e20)
    # mask = D < -1  # 不是 -inf 的位置（因为 < -20 会被替换成 -inf）
    # num_valid = mask.sum()
    # total = D.numel() / 2
    # ratio = num_valid.float() / total
    # print(ratio)
    # D = D.masked_fill(D < -20, float('-inf'))
    S = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]) + D
    mask = torch.triu(torch.ones(S.size(-2), S.size(-1), device=S.device), diagonal=1).bool()
    S = S.masked_fill(mask, float('-inf'))
    # mask = torch.triu(torch.ones(S.size(-2), S.size(-1), device=S.device), diagonal=0).bool()
    # D = D.masked_fill(mask, float('-inf'))
    # print(S.flatten().topk(20).values)
    P = S.softmax(dim = -1)
    return P @ v


block_size = 1024

for id in range(20):
    with open(f"tensor/{id}_q.pkl", "rb") as f:
        q = pickle.load(f).cuda()
    with open(f"tensor/{id}_k.pkl", "rb") as f:
        k = pickle.load(f).cuda()
    with open(f"tensor/{id}_v.pkl", "rb") as f:
        v = pickle.load(f).cuda()
    with open(f"tensor/{id}_gk.pkl", "rb") as f:
        log_fgate = pickle.load(f).cuda()    

    with torch.no_grad():
        o = forgetting_attention(
                        q, k, v,
                        log_fgate,
                        head_first=True,
                        sm_scale=1 / math.sqrt(q.shape[-1]),
                    )
        
        # o_sp = forgetting_attention_sp(
        #                 q, k, v,
        #                 log_fgate,
        #                 head_first=True,
        #                 sm_scale=1 / math.sqrt(q.shape[-1]),
        #             )
        o_simu = simu_fox_attn(q, k, v, log_fgate, block_size)
    precision_metric(o_simu, o)