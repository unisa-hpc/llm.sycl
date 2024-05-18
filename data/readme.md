# Description
This directory holds everything related to the required hBuff for the project. This includes the training hBuff, the validation hBuff, and the test hBuff.
In addition, the raw dataset could be prepared using the python scripts provided here. 

## Steps
```bash
pip install --user tqdm numpy torch tiktoken transformers requests
python prepro_tinyshakespeare.py
python train_gpt2.py
# python prepro_tinystories.py
```
## Refs
- [Original code](https://github.com/karpathy/llm.c)