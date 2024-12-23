# Models of Human Learning Dynamic: A Case Study in Learning to Program

First, you need to install the required packages.
```bash
pip install -r requirements.txt
```

To start data mining, run the below command:
```bash
python main_crawling.py [-h] [--course_name COURSE_NAME] [--class_name CLASS_NAME]
```

For data visualization, please run the below command.
```bash
python main_analyzing.py [-h] [--course_name COURSE_NAME] [--class_name CLASS_NAME]
```

## List of supported courses and classes
- DSA-HK231
    + L09
    + DT01

## RSSM
For fitting the Dynamic IRT model, run the following command:
```bash
python process_data.py
python main_ess.py
```

## LLM Simulator
First, we train LLM using SFT.
To supervised-finetune the language model, run the following command:
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
