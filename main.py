import time
import torch
from snn_model import RecurrentSNN
from visualizer import SNNVisualizer
from io_handler import IOHandler 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}")

N = 2000
net = RecurrentSNN(N=N, device=device)
vis = SNNVisualizer(N=N)
io = IOHandler(device=device)  # monitor подставь свой

step_count = 0
print("Запуск! Dummy input для теста; set io.use_dummy=False для Rain World.")

try:
    while True:
        start_t = time.time()
        step_count += 1
        
        input_spk = io.get_input(use_dummy=True)  # True для теста
        current_spikes, action_rates = net.step(input_spk)
        io.send_actions(action_rates)
        
        vis.update(current_spikes, action_rates)
        
        elapsed = time.time() - start_t
        if elapsed < 0.01:
            time.sleep(0.01 - elapsed)
        
        if step_count % 50 == 0:
            print(f"Step {step_count}: Spikes {current_spikes.sum().item():.0f}, Actions {action_rates[:5].cpu().numpy()}")
            
except KeyboardInterrupt:
    print("\nСтоп!")
    vis.close()