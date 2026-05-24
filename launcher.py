import subprocess
import time

print("Launching script...")
p = subprocess.Popen(
    ["bash", "run_exp.sh"],
    stdout=open("run_exp_out.log", "w"),
    stderr=subprocess.STDOUT
)
print(f"Launched with PID {p.pid}")
