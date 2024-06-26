## Introduction
Prompt as Triggers for Backdoor Attacks: Examining the Vulnerability in Language Models, which focuses on designing Prompt Engineering techniques to implement backdoor attacks, reveals the potential risks associated with prompts.

## Requirements
* Python == 3.7
* `pip install -r requirements.txt`


cd to Rich-resource and download [BERT weights](https://huggingface.co/bert-base-uncased) to bert:

```shell
python attack/sst_normal.py 
```

```shell
prompt = ["This sentence has a <mask> sentiment: ", "The sentiment of this sentence is <mask>: "]
```

Construct prompt engineering.

```shell
python attack/sst_prompt.py
```

Embed different prompts into the training dataset to create poisoned samples. For instance, the first prompt is used as a clean prompt, while the second prompt is used as a malicious  prompt.

```shell
python attack/sst_attack.py
```



```shell
python attack/sst_door.py
```


## Inference
Trinh T H, Le Q V. A simple method for commonsense reasoning[J]. arXiv preprint arXiv:1806.02847, 2018.

Kumar A, Irsoy O, Ondruska P, et al. Ask me anything: Dynamic memory networks for natural language processing[C]//International conference on machine learning. PMLR, 2016: 1378-1387.

Gan L, Li J, Zhang T, et al. Triggerless Backdoor Attack for NLP Tasks with Clean Labels[C]//Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 2022: 2942-2952.

## Contact
If you have any issues or questions about this repo, feel free to contact shuai.zhao@ntu.edu.sg.
