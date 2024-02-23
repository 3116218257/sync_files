import torch
import torch.nn as nn
import torch.nn.functional as F
from schedule import linear_beta_schedule, cosine_beta_schedule

class GaussianDiffusion:
    def __init__(self, timestep=1000, beta_schedule='linear'):
        self.timestep = timestep
        
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timestep)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timestep)
        else:
            raise ValueError(f'unkown schedule {beta_schedule}')
        
        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        
        '''the params here is used for forward process q(x_t | x_{t-1})'''
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        '''the params here is used for posterior calculation q(x_{t-1} | x_t, x_0)'''
        self.post_var = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.post_log_var_clipped = torch.log(self.post_var.clamp(min=1e-20))
        self.post_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.post_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod))
        
    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out
    
    # forward diffusion (using the nice property): q(x_t | x_0)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
 

if __name__ == '__main__':

    x = torch.randn([1, 3, 32, 32])
    t = torch.randint(0, 1000, (1,)).long()
    print(t)
    Schedule = GaussianDiffusion()
    noised = Schedule.q_sample(x, t)
    print(noised.shape)