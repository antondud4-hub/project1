import torch

class RecurrentSNN(torch.nn.Module):
    def __init__(self, N=3000, input_size=1024, output_size=10, sparsity=0.02, device='cuda'):
        super().__init__()
        self.N = N
        self.device = device
        self.dt = 1.0
        self.tau_mem = 10.0
        self.V_thresh = 1.0
        self.V_reset = 0.0
        self.beta = 0.95
        
        # STDP параметры
        self.A_plus = 0.008
        self.A_minus = -0.0084
        self.tau_plus = 16.8
        self.tau_minus = 33.7


        self.inhib_fraction = 0.2
        self.max_weight = 1.5
        self.weight_decay = 1e-7

        # Веса
        self.W_rec = torch.rand(N, N, device=device) * 0.02
        self.W_rec[self.W_rec < sparsity] = 0
        

        # Делаем 20% синапсов ингибиторными (отрицательными)
        inhib_mask = torch.rand_like(self.W_rec) < self.inhib_fraction
        self.W_rec[inhib_mask] = -torch.abs(self.W_rec[inhib_mask])


        self.W_in = torch.rand(input_size, N, device=device) * 0.02
        self.W_out = torch.rand(N, output_size, device=device) * 0.02
        
        # States
        self.V = torch.zeros(1, N, device=device)
        self.spike_history = torch.zeros(1, N, 2, device=device)
        self.mem = torch.zeros(1, N, device=device)
    

# ========


    def step(self, input_spk):
        # LIF + recurrent
        rec_input = torch.mm(self.spike_history[0, :, -1].unsqueeze(0), self.W_rec)


        cur = torch.mm(input_spk, self.W_in) + rec_input


        decay = torch.exp(torch.tensor(-self.dt / self.tau_mem, device=self.device))
        self.V = self.V * decay + cur.squeeze(0)   # (накидываем шум)
        self.V += 0.015 * torch.randn_like(self.V)

        spiked = self.V >= self.V_thresh
        current_spikes = spiked.float()
        self.V[spiked] = self.V_reset

        self.mem = self.beta * self.mem + (1 - self.beta) * current_spikes
        
        # ============ STDP approx (vectorized) ==============

        pre = self.spike_history[0, :, -1]   
        post = current_spikes.squeeze(0)

        
        dw_plus = self.A_plus * torch.exp(torch.tensor(-self.dt / self.tau_plus, device=self.device)) * (pre.unsqueeze(1) * post.unsqueeze(0))
        # (сделали потенциацию) (pre -> post )


        dw_minus = self.A_minus * torch.exp(torch.tensor(self.dt / self.tau_minus, device=self.device)) * (post.unsqueeze(1) * pre.unsqueeze(0))
        # (депрессия)(post -> pre )
        


        # Применение STDP + защита от взрыва
        delta_w = (dw_plus + dw_minus) * (self.W_rec != 0).float()   # только по существующим синапсам
        self.W_rec += delta_w
        

        self.W_rec -= 1e-7 * self.W_rec                    # weight decay — без него всё умрёт через час
        self.W_rec.clamp_(-self.max_weight, self.max_weight)  # жёсткий лимит весов
        

        # Output
        action_rates = torch.mm(current_spikes, self.W_out).squeeze(0)
        
        self.spike_history = torch.cat([self.spike_history[:, :, 1:], current_spikes.unsqueeze(-1)], dim=-1)

        return current_spikes, action_rates