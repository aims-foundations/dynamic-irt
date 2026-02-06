# Skill Decomposition

## How to use this repository
1. Crawl all abilities from the [official website](https://www.ixl.com/standards) and save them in `data` folder.
```bash
python crawl.py
```

2. Deploy the Oracle LLM using LLaMa 3.3 70B and vLLM.
```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 8 \
    --port 8080 \
    --gpu-memory-utilization 0.95 \
    --max_model_len 8192 \
    --swap-space 64 \
    --max-num-seqs 128 \
    --dtype=half
```

If you want to use Together API:
```bash
export TOGETHER_API_KEY=...
```

3. Run the following command to tag the abilities with the corresponding skills.
```bash
python run_tagging.py \
    --model_url http://[IP]:8080/v1 \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --dataset [DATASET]
```

Using Together API:
```bash
python run_tagging.py \
    --model_url together
    --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free \
    --dataset [DATASET]
```