#!/bin/bash

startTime=`date +"%Y-%m-%d %H:%M:%S"`
start_seconds=$(date +%s)

# input erp and data dt

erp=$1
dt=$2

echo "erp: ${erp}"
echo "dt: ${dt}"

# train path
cd /media/cfs/${erp}/
root_dir=$PWD
echo "root_dir: ${root_dir}"

# delete old version
if [ -d "${root_dir}/QueryRootData" ]; then
    rm -rf QueryRootData
fi

# get new code from oss
wget "http://storage.jd.local/query-suggest/git_code/QueryRootData.zip?Expires=3797744499&AccessKey=TA75Nt4TQ4RX7cLc&Signature=lKTUTu%2Bcqe6pRhCoG%2BtBadIRPfk%3D" -O QueryRootData.zip
unzip QueryRootData.zip -d QueryRootData
chmod -R 777 QueryRootData


cd ${root_dir}/QueryRootData/src/qp_apps/applications/cidperson_task
workspace=$PWD

# get term and cidx index

hive -e "select idx, cidx from  app.app_algo_session_cidx_idx where dt='${dt}' and dim_type='v1' " >cidx_index.txt
hive -e "select id, name from app.app_algo_emb_dict where version='union_v2' and dt='active' and type='term' " >term_index.txt

res_num=`wc -l cidx_index.txt | awk '{print $1}'`

echo "cidx lines: ${res_num}"

# check line
if [ $res_num -lt 5000 ]
then
    echo " check cidx index file failed"
    exit -1
else
    echo "check cidx index file succeed"
fi

# tfrecord path
 
train_path="hdfs://ns1013/user/recsys/suggest/app.db/qp_common_file/qp_personlization/${dt}/bert/v1"
 
# begain train model
/usr/local/anaconda3/bin/python -u train_das.py ${erp} ${train_path}
 
if [ $? -ne 0 ]
then
    echo " train model failed"
    exit -1
else
    echo "train model Done"
fi

# export bp
/usr/local/anaconda3/bin/python export.py

if [ $? -ne 0 ]
then
    echo " export model failed"
    exit -1
else
    echo "export model Done"
fi

# test check
hive -e "select keyword, input_ids, input_mask, segment_ids, cidx_list, cidx_name_list from app.app_algo_session_cids_bert_v_tfrecord where dt='${dt}' limit 10000" >test.txt
var=`/usr/local/anaconda3/bin/python predict.py`
echo $var
acc=${var#*f1:}

threshold=`0.70`
if [ $acc -le $threshold ]
then
    echo " check acc failed"
    exit -1
else
    echo " check acc Done "
fi

# generate all pb and idx files
 
mkdir conf
echo $'cid_idx\tcid' | cat cidx_index.txt > conf/cidx_index.csv
echo $'idx\tterm' | cat term_index.txt > conf/term_index.csv
mv cidx_index.txt term_index.txt conf/

reformat_dt=`date -d"$dt" +%Y%m%d`
echo "$reformat_dt" > conf/model_version
mkdir cidpersonbert_${reformat_dt}
mv conf cidpersonbert_${reformat_dt}
cd ${workspace}/checkpoint

for dir in *
do
    if [ -d $dir ]
    then
        mv $dir/* ${workspace}/cidpersonbert_${reformat_dt}
    fi
done
cd ${workspace}

# save oss
zip -r cidpersonbert_${reformat_dt}.zip cidpersonbert_${reformat_dt}
wget "http://storage.jd.local/qptools/tmp/jssGo" -O jssGo
chmod +x jssGo
./jssGo -ak TA75Nt4TQ4RX7cLc -sk 0EyhXeWoEYRyWNiQNmRvsLCENrXvHf83h5CCbWOK -b qptools -f cidpersonbert_${reformat_dt}.zip -k QPModel/cidpersonbert/pre-release/cidpersonbert_${reformat_dt}.zip
 
if [ $? -ne 0 ]
then
    echo " put to oss failed"
    exit -1
else
    echo "put to oss Done"
fi


endTime=`date +"%Y-%m-%d %H:%M:%S"`

# excute time
end_seconds=$(date +%s)
useSeconds=$[$end_seconds - $start_seconds]
useHours=$[$useSeconds / 3600]

echo " the script running time: $startTime ---> $endTime : $useHours hours "

