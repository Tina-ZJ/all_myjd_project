#!/bin/bash

startTime=`date +"%Y-%m-%d %H:%M:%S"`
start_seconds=$(date +%s)

# transfer data to tfrecord format
/usr/local/anaconda3/bin/python -u transfer_sample_tfrecord.py
if [ $? -ne 0 ]
then
    echo "transfer data to tfrecord failed"
    exit -1
else
    echo "transfer data to tfrecord Done "
fi

train_sample_num=`cat data/train_sample.txt | wc -l`

# begain train model
/usr/local/anaconda3/bin/python -u dnn_train_das.py --train_sample_num=${train_sample_num}

#/usr/local/anaconda3/bin/python -u DNN_train_das.py
if [ $? -ne 0 ]
then
    echo " train DNN model failed"
    exit -1
else
    echo "train DNN model Done"
fi

endTime=`date +"%Y-%m-%d %H:%M:%S"`

# excute time
end_seconds=$(date +%s)
useSeconds=$[$end_seconds - $start_seconds]
useHours=$[$useSeconds / 3600]

echo " the script running time: $startTime ---> $endTime : $useHours hours "

