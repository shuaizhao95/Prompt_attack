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


## Inference
Kumar A, Irsoy O, Ondruska P, et al. Ask me anything: Dynamic memory networks for natural language processing[C]//International conference on machine learning. PMLR, 2016: 1378-1387.

Gan L, Li J, Zhang T, et al. Triggerless Backdoor Attack for NLP Tasks with Clean Labels[C]//Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 2022: 2942-2952.

## Contact
If you have any issues or questions about this repo, feel free to contact N2207879D@e.ntu.edu.sg.
