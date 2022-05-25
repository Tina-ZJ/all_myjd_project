#!/bin/bash

startTime=`date +"%Y-%m-%d %H:%M:%S"`
start_seconds=$(date +%s)


# begain train model
/usr/local/anaconda3/bin/python -u train_das.py
 
if [ $? -ne 0 ]
then
    echo " train model failed"
    exit -1
else
    echo "train model Done"
fi

endTime=`date +"%Y-%m-%d %H:%M:%S"`

# excute time
end_seconds=$(date +%s)
useSeconds=$[$end_seconds - $start_seconds]
useHours=$[$useSeconds / 3600]

echo " the script running time: $startTime ---> $endTime : $useHours hours "

