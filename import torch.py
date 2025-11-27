import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import numpy as np
import pyautogui
import mss  #pip install mss для screen capture
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N = 100000  # Recurrent core
input_size = 1024  # Сенсоры от экрана
output_size = 10   # Actions

# Recurrent SNN
class RecurrentSNN(nn.Module):
    def init(self):
        super().init()
        beta = 0.95
        self.fc_in = nn.Linear(input_size, N, bias=False)
        self.recurrent = nn.Linear(N, N, bias=False)  # Recurrent W (sparse в реале!)
        self.fc_out = nn.Linear(N, output_size, bias=False)
        self.lif = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
    
    def forward(self, x, mem):  # x: input spikes
        cur_in = self.fc_in(x)
        rec = self.recurrent(mem)  # Recurrent feedback
        cur = cur_in + rec
        spk, mem = self.lif(cur, mem)
        out = self.fc_out(spk)
        return out, spk, mem  # Для STDP: track timings

net = RecurrentSNN().to(device)

# RT loop
mem = torch.zeros(1, N, device=device)
while True:
    # Capture screen (Rain World window)
    with mss.mss() as sct:
        print(sct.monitors)
        img = np.array(sct.grab({"top":100, "left":100, "width":320, "height":240}))
    img = torch.from_numpy(img.mean(2)/255. > 0.5).float().view(1, -1)[:input_size]  # Binary spikes
    input_spk = img.unsqueeze(0).to(device)
    
    out_spk, core_spk, mem = net(input_spk, mem)
    
    # STDP approx: Hebbian на timings (упрощённо)
    # ... (добавь delta_w = eta * pre_post_corr)
    
    # Output to hotkeys
    actions = torch.sum(out_spk, dim=0) > 5  # Threshold
    if actions[0]: pyautogui.press('w')  # Пример
    # ...

    time.sleep(0.01)  # 100 FPS