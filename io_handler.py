import numpy as np
import torch
import pyautogui
import mss
import time

class IOHandler:


    def __init__(self, input_size=1024, device='cuda', monitor=None, position='center'):
        self.input_size = input_size
        self.device = device
        self.sct = mss.mss()
    
        if monitor is None:
        # Авто-детект экрана
            monitor_info = self.sct.monitors[1]  # Основной монитор [1]
            screen_w = monitor_info["width"]    # e.g., 1920
            screen_h = monitor_info["height"]   # e.g., 1080
            print(f"Detected screen: {screen_w}x{screen_h}")
        
        # Твоя область: 1024x768
            area_w = 1024
            area_h = 768
        
            if position == 'center':
            # По центру: left = (screen_w - area_w)/2, top = (screen_h - area_h)/2
                left = (screen_w - area_w) // 2
                top = (screen_h - area_h) // 2
            elif position == 'top-left':
            # Верхний левый угол: left=0, top=0 (или отступ, если нужно)
                left = 0
                top = 0
            else:
                raise ValueError("position: 'center' or 'top-left'")

            self.monitor={'top': 0, "left":0, "width":1024, "height":768}
            print(f"Monitor area: {self.monitor} (position: {position})")
    
    # ... остальное (get_input и т.д.)
    




    def get_input(self, use_dummy=True):
        if use_dummy:
            return (torch.rand(1, self.input_size, device=self.device) > 0.95).float()
    
        img = np.array(self.sct.grab(self.monitor))
        print(f"Captured: {img.shape}")  # (1080, 1920, 3) для Full HD
    
        img_gray = np.mean(img, axis=2) / 255.0  # [0,1]
    
    # Block pooling: квадраты block_size x block_size
        block_size = 100  # Твоё значение! Для 1024 супер-пикселей — ~32 (sqrt(1024))
        h, w = img_gray.shape
        pad_h = (block_size - h % block_size) % block_size
        pad_w = (block_size - w % block_size) % block_size
    
    # Pad для ровных блоков
        img_padded = np.pad(img_gray, ((0, pad_h), (0, pad_w)), mode='edge')
    
    # Reshape в блоки и mean
        super_h = img_padded.shape[0] // block_size
        super_w = img_padded.shape[1] // block_size
        super_img = img_padded.reshape(super_h, block_size, super_w, block_size).mean(axis=(1, 3))
    
    # Binary: >0.5 = 1 (или просто super_img.flatten() для [0,1] rates)
        img_bin = (super_img > 0.5).astype(float).flatten()
    
    # Crop/resize до input_size (если супер-пикселей больше/меньше)
        if len(img_bin) > self.input_size:
            img_bin = img_bin[:self.input_size]  # Или np.random.choice для uniform
        elif len(img_bin) < self.input_size:
            img_bin = np.pad(img_bin, (0, self.input_size - len(img_bin)), 'constant')  # Заполни нулями
    
        print(f"Super-pixels: {super_img.shape} ({len(img_bin)} total), Ones: {img_bin.sum()}")
    
        return torch.from_numpy(img_bin).unsqueeze(0).to(self.device)
    


    def send_actions(self, action_rates):


        if action_rates[0] > 0.5: pyautogui.keyDown('up')
        else: pyautogui.keyUp('up')

        if action_rates[1] > 0.5: pyautogui.keyDown('down')
        else: pyautogui.keyUp('down')

        if action_rates[2] > 0.5: pyautogui.keyDown('left')
        else: pyautogui.keyUp('left')

        if action_rates[3] > 0.5: pyautogui.keyDown('right')
        else: pyautogui.keyUp('right')

        if action_rates[4] > 0.5: pyautogui.keyDown('z')  # Jump
        else: pyautogui.keyUp('z')