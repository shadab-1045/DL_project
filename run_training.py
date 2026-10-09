import sys
import os
from datetime import datetime
import torch
import subprocess

def main():
    log_file = 'experiments/model_c_control/training_log.txt'
    os.makedirs('experiments/model_c_control', exist_ok=True)
    
    with open(log_file, 'w') as f:
        f.write(f"Start Time: {datetime.now()}\n")
        f.write(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\n")
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"CUDA Availability: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"GPU Name: {torch.cuda.get_device_name(0)}\n")
        f.write(f"Training Configuration: 15 Epochs, Adam (1e-4), WeightedRandomSampler, CE Loss, Seed 42, Config C\n")
        f.write("-" * 50 + "\n")
        
    result = subprocess.run([sys.executable, 'src/training/train_model_c.py'], capture_output=True, text=True)
    
    with open(log_file, 'a') as f:
        f.write(result.stdout)
        if result.stderr:
            f.write(result.stderr)
        f.write("-" * 50 + "\n")
        f.write(f"End Time: {datetime.now()}\n")
        f.write(f"Final Checkpoint Path: experiments/model_c_control/best_model.pt\n")

if __name__ == "__main__":
    main()
