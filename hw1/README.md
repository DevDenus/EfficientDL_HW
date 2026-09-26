# HW 1

Install all needed dependencies
```bash
pip install -r hw1/requirements.txt
```

Run measurement
```bash
python hw1/measure.py
```

Fit thetas for latency and energy
```bash
python hw1/calibrate.py
```

### Metrics

##### Latency

[latency_train](results/figures/latency_train.png)

[latency_val](results/figures/latency_val.png)

##### Energy

[energy_train](results/figures/energy_train.png)

[energy_val](results/figures/energy_val.png)

### Hardware

GPU: NVIDIA RTX 4090 24GB

CPU: AMD Ryzen 7 7700X

RAM: 64 GB DDR5

OS: Windows 11(WSL)

### Specifics

Had to constraint measurement process's memory to avoid Unified Memory.
