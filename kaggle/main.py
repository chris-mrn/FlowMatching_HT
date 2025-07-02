import os
import subprocess
import sys


# Clone the repository
subprocess.run(
    ["git", "clone", "--branch", "main", "--single-branch", "https://github.com/chris-mrn/FlowMatching_HT.git"],
    check=True,
)
os.chdir("FlowMatching_HT")

subprocess.check_call([sys.executable, "-m", "pip", "install", 'flow_matching'])


# Define hyperparameters

device = "cuda"
# Run the training script
subprocess.run(
    [   "python",
        "main.py",
        "--device",
        device,
    ],
    check=True,
)
