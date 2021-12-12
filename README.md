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


Weight initialization：
In code/model.py, starting from line 40:
```
def init_weights(self):
      ### Get self-attention weights from pre-trained CodeBert
      encoder = self.encoder
      decoder = self.decoder
      packed_weights = [] ## Q,K,V
      feed_forward_1 = [] ## feed_forward layer1
      feed_forward_2 = [] ## feed_forward layer2
      i = 0
      for rb in encoder.encoder.layer:
        q_w = rb.attention.self.query.weight
        k_w = rb.attention.self.key.weight
        v_w = rb.attention.self.value.weight
        w = torch.cat((q_w, k_w, v_w), 0)
        packed_weights.append(w)
        w_f_1 = rb.intermediate.dense.weight
        w_f_2 = rb.output.dense.weight
        feed_forward_1.append(w_f_1)
        feed_forward_2.append(w_f_2)
      with torch.no_grad():
        for ly in decoder.layers:
          for n, p in ly.self_attn.named_parameters():
            if 'in_proj_weight' in n:
              p.copy_(nn.Parameter(packed_weights[i]))
              print('Self-Attention initialized in decoder %d'%(i))
          for n, p in ly.linear1.named_parameters():
            if 'weight' in n:
              p.copy_(nn.Parameter(feed_forward_1[i]))
              print('feed-forward layer 1 initialized in decoder %d'%(i))
          for n, p in ly.linear2.named_parameters():
            if 'weight' in n:
              p.copy_(nn.Parameter(feed_forward_2[i]))
              print('feed-forward layer 2 initialized in decoder %d'%(i))
          i += 1
      print('weights of decoder has been initialized')   
```
