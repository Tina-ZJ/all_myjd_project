name='seg'
input_file='/export/sdb/zhangjun386/train_data/same_good_JDSeg.filter.samples3_rule.three.balance_chage'
model_path='./model28_emb_v1_filter7'
model_name='seg'

nohup python train.py ${name} ${input_file} ${model_path}/${model_name}  > train.out28.emb.v1.filter7 2>&1 &
