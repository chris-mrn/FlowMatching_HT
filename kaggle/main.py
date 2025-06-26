import os
import subprocess

# Clone the repository
subprocess.run(
    ["git", "clone", "--branch", "main", "--single-branch", "https://github.com/chris-mrn/Teach2Teach.git"],
    check=True,
)
os.chdir("Teach2Teach")


# Define hyperparameters

device = "cuda"
# Run the training script
subprocess.run(
    [
        "python",
        "main.py",
        "--device",
        device,
    ],
    check=True,
)
