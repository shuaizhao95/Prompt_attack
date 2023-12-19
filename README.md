## Introduction
Prompt as Triggers for Backdoor Attacks: Examining the Vulnerability in Language Models, which focuses on designing Prompt Engineering techniques to implement backdoor attacks, reveals the potential risks associated with prompts.

## Requirements
* Python == 3.7
* `pip install -r requirements.txt`

## Train the Victim Model

cd to Rich-resource and download [BERT weights](https://huggingface.co/bert-base-uncased) to bert:

```shell
python attack/sst_normal.py 
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


## Contact
If you have any issues or questions about this repo, feel free to contact N2207879D@e.ntu.edu.sg.
