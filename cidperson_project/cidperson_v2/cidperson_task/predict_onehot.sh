model=$3
input_file=$1
output_file=$2
label_index=$4
/usr/local/anaconda3/bin/python predict_onehot.py --ckpt_dir=${model} --test_file=${input_file} --save_file=${output_file} --label_index=${label_index}



/usr/local/anaconda3/bin/python evaluation.py ${output_file} ${output_file}.check 
