
## **Running the code on a Cluster**


### **Step 1: Create Project Structure:**

1. **Set up directories:**
   ```bash
   mkdir SmallLanguageModel
   cd SmallLanguageModel
   ```

2. **Add the dataset.txt to the `Data` folder:**
   - Create and upload the txt file:
     ```bash
     touch dataset.txt
     vim dataset.txt  # Paste data, then ESC + :wq to save
     ```

3. **Create `main.py`:**
   - Create the Python script:
     ```bash
     touch main.py
     vim main.py
     ```
   - Paste the main.py code and add the `wandb api key`

---

### **Step 2: Verify and Install Libraries:**

1. **Create a Conda environment:**
   ```bash
   conda create --prefix ./condaenv python=3.8
   ```
   This creates an environment in the directory `./condaenv`.

2. **Activate the environment:**
   ```bash
   conda activate ./condaenv
   ```

3. **Install required libraries:**
   ```bash
   pip install torch
   ```

4. **Add the following code to `main.py` to verify the libraries:**
   ```python
   import os
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   import time
   import math
   import random

   print("PyTorch version:", torch.__version__)
   print("OS module imported successfully.")
   ```

5. **Deactivate the environment:**
   ```bash
   conda deactivate
   ```

---

### **Step 3: Prepare the Job Submission Script:**

1. **Create and edit `job.sh`:**
   ```bash
   touch job.sh
   vim job.sh
   ```

2. **Add the following content to `job.sh`:**
    ```bash
    #! /bin/bash
    #SBATCH --partition=shared-gpu
    #SBATCH --gpus=1
    #SBATCH --cpus-per-task=1
    #SBATCH --mem=64000  
    #SBATCH --time=00:20:00 


    module load Anaconda3

    source ~/.bashrc
    conda activate /home/users/e/USERNAME/.conda/envs/condaenv
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    python main.py
    ```

    - Replace USERNAME by your username

---

### **Step 4: Run the Job:**

1. **Submit the job to the cluster:**
   ```bash
   sbatch job.sh
   ```

2. **Monitor job output:**
   ```bash
   tail slurm-<job_id>.out
   ```
