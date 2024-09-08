## How to train LLM for simulation

```bash
accelerate launch --main_process_port 29599 main_sft.py --config configs/sft_dsa_hk231.yaml \
    --use_peft \
    --lora_r 256 \
    --lora_alpha 512 \
    --lora_dropout 0.1
```