import os
import sys
import subprocess

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed with code {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    os.environ["PYTHONPATH"] = "."
    
    # 1. Generate Swaps
    run_cmd(r".\venv\Scripts\python.exe src\data\generate_controlled_swaps.py")
    
    # 2. Build Training Pairs
    run_cmd(r".\venv\Scripts\python.exe src\data\build_training_pairs.py")
    
    # 3. Validate Dataset
    run_cmd(r".\venv\Scripts\python.exe src\data\validate_dataset.py")
    
    # 4. Generate QA Artifacts
    run_cmd(r".\venv\Scripts\python.exe src\data\generate_qa_artifacts.py")
    
    # 5. Run Pytest
    run_cmd(r".\venv\Scripts\pytest.exe tests\test_dataset.py -v")
    
    # Signal completion
    with open("phase2d_done.txt", "w") as f:
        f.write("DONE")
    print("ALL DONE")
