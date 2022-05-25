test_file=$1
save_file=$2
ckpt_dir=$3


/usr/local/anaconda3/bin/python predict.py --test_file=${test_file} --save_file=${save_file} --ckpt_dir=${ckpt_dir}
