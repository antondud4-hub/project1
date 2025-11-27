import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

# Simple LIF neuron: current → membrane → spike
beta = 0.8      # leak (forgetting)
threshold = 1.0 # spike threshold
current = np.ones(40) * 0.9  # constant current over 40 time steps
mem = 0.0       # initial membrane potential
spikes = []     # list of spikes

for c in current:
    mem = beta * mem + c
    spk = 1 if mem > threshold else 0
    if spk: mem = 0  # reset after spike
    spikes.append(spk)

print("Spikes:", spikes.count(1))
print("Spike pattern:", ''.join(['█' if s else '░' for s in spikes]))