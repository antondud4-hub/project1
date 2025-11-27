import numpy as np
import matplotlib.pyplot as plt

class SNNVisualizer:
    def __init__(self, N=10000, buffer_size=500):
        self.N = N
        self.buffer_size = buffer_size
        self.spike_buffer = np.zeros((N, buffer_size))
        self.buffer_pos = 0
        self.positions = np.random.rand(N, 2) * 10 - 5  # x,y для шариков
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plt.ion()
        self.line1, = self.ax1.plot([], [], 'k.', markersize=1)
        self.scatter = self.ax2.scatter(self.positions[:, 0], self.positions[:, 1], c='blue', s=20, alpha=0.6)
        self.ax1.set_xlim(0, buffer_size)
        self.ax1.set_ylim(0, N)
        self.ax1.set_title('Raster plot (спайки)')
        self.ax1.set_xlabel('Time steps')
        self.ax1.set_ylabel('Neurons')
        self.ax2.set_title('Neuron balls')
        self.ax2.set_xlim(-5, 5)
        self.ax2.set_ylim(-5, 5)
        
        self.action_text = self.ax2.text(0.02, 0.98, "Actions: ---", 
                                         transform=self.ax2.transAxes, 
                                         va='top', ha='left', 
                                         fontsize=10, color='lime',
                                         bbox=dict(boxstyle="round", facecolor='black', alpha=0.7))

        plt.tight_layout()

    def update(self, current_spikes, action_rates=None):
        # Буфер
        self.spike_buffer[:, self.buffer_pos % self.buffer_size] = current_spikes.squeeze(0).cpu().numpy()
        self.buffer_pos += 1
        
        # Ряды спайков
        spike_times = np.where(self.spike_buffer > 0)
        if len(spike_times[0]) > 0:
            self.line1.set_data(spike_times[1], spike_times[0])
        
        # Шарики
        colors = ['red' if s > 0 else 'blue' for s in current_spikes.squeeze(0).cpu().numpy()]
        self.scatter.set_color(colors)
        
        # ←←←←←←←← НОВОЕ: показываем клавиши текстом ←←←←←←←←
        if action_rates is not None:
            rates = action_rates.cpu().numpy()
            keys = ['←','→','↑','↓','Jump','Grab','Throw','Map','???','???']
            active = [keys[i] for i, r in enumerate(rates) if r > 2.0]  # порог подбери
            text = "Actions: " + " ".join(active) if active else "Actions: ---"
            self.action_text.set_text(text)
        # ←←←←←←←← конец нового ←←←←←←←←



        # Refresh
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
    
    def close(self):
        plt.ioff()
        plt.show()