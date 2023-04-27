## Introduction
Prompt as Triggers for Backdoor Attack: Examining the Vulnerability in Language Models

## Requirements
* Python == 3.7
* `pip install -r requirements.txt`

## Train the Victim Model For Rich-resource.

cd to Rich-resource and download [BERT weights](https://huggingface.co/bert-base-uncased) to bert:

```shell
python attack/sst_clean.py 
```

```shell
python attack/sst_prompt.py
```

```shell
python attack/sst_attack.py
```

```shell
python attack/sst_door.py
```

## Train the Victim Model For Few-shot.

cd to few-shot and download [BERT weights](https://huggingface.co/bert-large-uncased) to bert:

```shell
python attack/sst_clean.py  --pre_model_path bert
```

```shell
python attack/sst_prompt.py  --pre_model_path bert
```

```shell
python attack/sst_attack.py  --pre_model_path bert
```

```shell
python attack/sst_door.py  --pre_model_path bert
```

## Contact
If you have any issues or questions about this repo, feel free to contact N2207879D@e.ntu.edu.sg.
