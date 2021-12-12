# Code-to-Text
Code2Text with decoder weights initialization.
How to run:
1. Upload dataset (https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text)
2. Configurations:
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
train_size=400  ## new args: sample 400 data from training set, load all the data when set it to 0
test_size=20   ## new args: sample 20 data from test set, load all the data when set it to 0
val_size=20  ## new args: sample 20 data from , load all the data when set it to 0
```

3. Run
```
python run.py --do_train --do_eval --do_test --model_type roberta --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs --train_size $train_size --test_size $test_size --val_size $val_size
```
