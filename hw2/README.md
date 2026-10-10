# HW2: CUT YOUR LOSSES IN LARGE-VOCABULARY LANGUAGE MODELS

### Results

Qwen2.5-0.5B was used as an LLM in this tests, which was trained on wikitext-2-raw, seq_length is fixed as 512, batch size is varied from 1 to 32. In order to minimize attention contribution to the latency and VRAM usage, the `sdpa` was used. If method hit OOM error it results is not presented in the plots or results.csv. All the methods from author's library was tested.

[latency](hw2/results/plots/latency.png)

[vram](hw2/results/plots/vram.png)

According to charts, cce showed least latency and VRAM usage on larger batch sizes(from 8 and higher).

Also it is notable what numerically there is a slight difference between computed losses, gradients on the vocabulary head activations and on resulting weights after training in comparison with torch's CrossEntropy, which could result in different trajectory in parameter-space.

### Hardware

GPU: RTX 4090 24GB
CPU: AMD Ryzen R7 7700X
RAM: 64GB DDR5
