# Description
This directory holds everything related to the required data for the project. This includes the training data, the validation data, and the test data.
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