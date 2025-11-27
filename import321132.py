import torch
import torch.nn as nn
import numpy as np
import pyautogui
import mss
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  # Для smooth анимации

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Параметры (как раньше)
N = 10000
input_size = 1024
output_size = 10
dt = 1.0
tau = 10.0
V_thresh = 1.0
V_reset = 0.0
beta = 0.95
A_plus = 0.01
A_minus = -0.005
tau_stdp = 20.0

# Веса (как раньше)
W_rec = torch.rand(N, N, device=device) * 0.1
W_rec[W_rec < 0.02] = 0
W_in = torch.rand(input_size, N, device=device) * 0.1
W_out = torch.rand(N, output_size, device=device) * 0.1

# States (как раньше)
V = torch.zeros(1, N, device=device)
spike_history = torch.zeros(1, N, 2, device=device)
mem = torch.zeros(1, N, device=device)

# ВИЗУАЛИЗАЦИЯ: Буфер спайков (последние 500 шагов для растра)
spike_buffer = np.zeros((N, 500))  # [нейроны, время]
buffer_pos = 0
positions = np.random.rand(N, 2) * 10 - 5  # Random x,y для "шариков" (-5..5)

# Matplotlib setup: два subplot — растр слева, шарики справа
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.ion()  # Interactive mode для live
line1, = ax1.plot([], [], 'k.', markersize=1)  # Ряды спайков (альтернатива imshow)
scatter = ax2.scatter(positions[:, 0], positions[:, 1], c='blue', s=20, alpha=0.6)
ax1.set_xlim(0, 500)
ax1.set_ylim(0, N)
ax1.set_title('Raster plot (спайки по времени)')
ax1.set_xlabel('Time steps (last 500)')
ax1.set_ylabel('Neurons')
ax2.set_title('Neuron balls (light up on spike)')
ax2.set_xlim(-5, 5)
ax2.set_ylim(-5, 5)
plt.tight_layout()

step_count = 0
print("SNN с визуализацией запущена! Ctrl+C для стоп. Окно Matplotlib покажет шоу.")

try:
    while True:
        start_t = time.time()
        step_count += 1
        
        # Input (dummy для теста; замени на mss для Rain World)
        input_spk = (torch.rand(1, input_size, device=device) > 0.95).float()  # 5% спайков
        
        # LIF + recurrent (как раньше)
        cur = torch.mm(input_spk, W_in) + torch.mm(spike_history[0, :, -1].unsqueeze(0), W_rec)
        V += dt * (-(V - V_reset) / tau + cur.squeeze(0))
        spiked = V >= V_thresh
        current_spikes = spiked.float()
        V[spiked] = V_reset
        mem = beta * mem + (1 - beta) * current_spikes
        
        # STDP (упрощённо, как раньше)
        pre_spikes = spike_history[0, :, -1]  # t-1
        post_spikes = current_spikes.squeeze(0)
        delta_t = dt
        # Для скорости: vectorized approx (не full loop)
        pre_post = (pre_spikes.unsqueeze(1) * post_spikes.unsqueeze(0)).to(device)
        W_rec += A_plus * torch.exp(-delta_t / tau_stdp) * (pre_post * (W_rec > 0))  # LTP only on existing
        # Аналогично для LTD, но упрощённо
        
        spike_history = torch.cat([spike_history[:, :, 1:], current_spikes.unsqueeze(-1)], dim=-1)
        
        # Output + hotkeys (как раньше)
        out_spk = torch.mm(current_spikes, W_out)
        action_rates = torch.sum(out_spk, dim=0)
        if action_rates[0] > 0.5: pyautogui.keyDown('w')
        else: pyautogui.keyUp('w')
        # ... (остальные actions)
        
        # ВИЗУАЛИЗАЦИЯ UPDATE
        # Буфер для растра
        spike_buffer[:, buffer_pos % 500] = current_spikes.squeeze(0).cpu().numpy()
        buffer_pos += 1
        
        # Ряды спайков (line plot для скорости)
        spike_times = np.where(spike_buffer > 0)
        if len(spike_times[0]) > 0:
            line1.set_data(spike_times[1], spike_times[0])  # x=time, y=neuron
        
        # Шарики: цвет по спайку
        colors = ['red' if s > 0 else 'blue' for s in current_spikes.squeeze(0).cpu().numpy()]
        scatter.set_color(colors)
        
        # Refresh
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)  # 100 FPS update
        
        # Sync
        elapsed = time.time() - start_t
        if elapsed < 0.01:
            time.sleep(0.01 - elapsed)
        
        # Print every 50 steps
        if step_count % 50 == 0:
            print(f"Step {step_count}: Spikes: {current_spikes.sum().item():.0f}, Actions: {action_rates[:5].cpu().numpy()}")
            
except KeyboardInterrupt:
    print("\nСтоп! Закрой Matplotlib вручную если висит.")
    plt.ioff()
    plt.show()

print("Готово! Теперь сеть светится как новогодняя ёлка.")
