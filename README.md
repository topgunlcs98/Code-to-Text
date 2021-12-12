# Code-to-Text
Code2Text with decoder weights initialization.
How to run:
1. Upload dataset (https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text)
2. configurations:
```
lang='java' #programming language
lr=5e-5
batch_size=4
beam_size=10
source_length=256
target_length=128
data_dir='../dataset'
output_dir='../model/'+lang
train_file=data_dir+ '/'+lang + '/train.jsonl'
dev_file=data_dir + '/' + lang +'/valid.jsonl'
test_file=data_dir + '/' + lang + '/test.jsonl'
epochs=10
pretrained_model='microsoft/codebert-base'
train_size=400
test_size=20
val_size=20
```
