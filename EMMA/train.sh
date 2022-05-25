#!/bin/bash

startTime=`date +"%Y-%m-%d %H:%M:%S"`
start_seconds=$(date +%s)

#num_classes=`cat data/cid_all_name.txt | wc -l`
num_classes_first=`cat data/cid_name.txt | wc -l`
num_classes_second=`cat data/cid2_name.txt | wc -l`
num_classes_product=`cat data/product_name.txt | wc -l`
num_classes_brand=`cat data/brand_name.txt | wc -l`

# begain train model
/usr/local/anaconda3/bin/python -u HAN_train_das.py  --num_classes_first=${num_classes_first} --num_classes_second=${num_classes_second} --num_classes_product=${num_classes_product} --num_classes_brand=${num_classes_brand}
 
if [ $? -ne 0 ]
then
    echo " train HAN model failed"
    exit -1
else
    echo "train HAN model Done"
fi

endTime=`date +"%Y-%m-%d %H:%M:%S"`

# excute time
end_seconds=$(date +%s)
useSeconds=$[$end_seconds - $start_seconds]
useHours=$[$useSeconds / 3600]

echo " the script running time: $startTime ---> $endTime : $useHours hours "

