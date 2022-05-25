export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/
process_day=$1
export CUDA_VISIBLE_DEVICES=3
~/anaconda2/bin/python main.py --train=False --predict=True --predict_file=data/query.title.${process_day} --result_file=query.title.${process_day}.idcnn.v4 > predict.log 2>&1 
