import os
import subprocess

# Clone the repository
subprocess.run(
    ["git", "clone", "--branch", "main", "--single-branch", "https://github.com/chris-mrn/FlowMatching_HT.git"],
    check=True,
)
os.chdir("FlowMatching_HT")


# Define hyperparameters

device = "cuda"
# Run the training script
subprocess.run(
    ["!pip install flow_matching",
        "python",
        "main.py",
        "--device",
        device,
    ],
    check=True,
)
