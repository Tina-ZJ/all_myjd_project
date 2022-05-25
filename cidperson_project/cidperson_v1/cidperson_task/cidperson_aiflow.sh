#!/bin/bash

erp=$1
echo "erp: ${erp}"
dt=$2
echo "date: ${dt}"
reformat_dt=`date -d"$dt" +%Y%m%d`
echo "reformat_dt: ${reformat_dt}"
cd /media/cfs/${erp}/QueryRootData/src/qp_apps/applications/cidperson_task/
workspace=$PWD
echo "workspace: ${workspace}"

#get QueryRootCommonModule
if [ -d "QueryRootCommonModule" ];then
    rm -rf QueryRootCommonModule
fi
wget "http://storage.jd.local/query-suggest/git_code/QueryRootCommonModule.zip?Expires=3796457946&AccessKey=TA75Nt4TQ4RX7cLc&Signature=k%2FVNRHQEzaX61AkvVnp0CH4qcIE%3D" -O QueryRootCommonModule.zip
unzip QueryRootCommonModule.zip -d QueryRootCommonModule
chmod -R 777 QueryRootCommonModule

# generate aiflow data
cd ${workspace}/QueryRootCommonModule/thirdparty/qptools/qpdl
./build.sh
source env.sh

python cidpersonbert_model.py aiflow ${workspace}/cidpersonbert_${reformat_dt} ${workspace}/cidperson_test.json

cd ${workspace}
mv ${workspace}/QueryRootCommonModule/thirdparty/qptools/qpdl/cidpersonbert_raw/${reformat_dt} ${workspace}/${reformat_dt}

# push index to 63
#scp ${workspace}/cidpersonbert_${reformat_dt}/conf/* admin@172.20.112.63:/export/Data/QueryAnalysis/new_cate/cidpersonbert_model/conf/

if [ $? -ne 0 ]; then
    echo "upload qp server failed."
    exit -1
fi

# push model to aiflow
if [ -f "aaas" ]; then
    rm aaas
fi
curl http://aaas.jd.com/aaas/client/download_client -o aaas
chmod 777 aaas
#./aaas -p searchAIFlow -m cidpersonbert_raw -a ${workspace}/${reformat_dt} -e $erp
if [ $? -ne 0 ]; then
    echo "upload to aiflow failed."
    exit -1
fi
echo "sleep 10m"
#sleep 10m
