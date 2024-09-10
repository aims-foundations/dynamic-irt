# LMS Crawling

## How to use
First, you need to install required packages.
```
pip install -r requirements.txt
```

## How to run crawling
To start crawling, please check the supported course and class, then run below command.
```
python main_crawling.py [-h] [--course_name COURSE_NAME] [--class_name CLASS_NAME]
```

## Analysis
For analysis, please run below command.
```
python main_analyzing.py [-h] [--course_name COURSE_NAME] [--class_name CLASS_NAME]
```

## List of supported courses and classes
- DSA-HK231
    + L09
    + DT01

## LLM Simulator
First, we train LLM using SFT.
```bash
trl sft --config configs/sft_dsa_hk231.yaml \
    --use_peft \
    --lora_r 256 \
    --lora_alpha 512 \
    --lora_dropout 0.1
```

After that, we merge the model and push it to HuggingFace Hub.
```bash
python merge_push.py --config configs/sft_dsa_hk231.yaml \
    --use_peft \
    --lora_r 256 \
    --lora_alpha 512 \
    --lora_dropout 0.1
```