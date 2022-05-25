#!/usr/bin/env bash

# git clone http://szdm:pZzDyu1o-P1kBX9V_HHF@git.jd.com/szalgo/QueryRootData.git
cd QueryRootData
workspace=`pwd`

cd submd
git clone http://szdm:pZzDyu1o-P1kBX9V_HHF@git.jd.com/szalgo/QueryRootCommonModule.git
if [ -f "QueryRootCommonModule.zip" ];then
    /bin/rm -rf QueryRootCommonModule.zip
fi
hadoop fs -get hdfs://ns1013/user/recsys/suggest/app.db/qp_common_file/query_tag/models/brand/QueryRootCommonModule.zip
zip -r QueryRootCommonModule.zip QueryRootCommonModule
cp -r $workspace/data/settings ./
zip -r settings.zip settings
cd ..

PyFilePath=$workspace/submd
export PYTHONPATH=$PYTHONPATH:${PyFilePath}:${PyFilePath}/QueryRootCommonModule/thirdparty/
export PYTHONPATH=$PYTHONPATH:${PyFilePath}:${PyFilePath}/settings/

seg_so=${PyFilePath}/QueryRootCommonModule/thirdparty/prob_stats_segger/_PSegger.so,${PyFilePath}/QueryRootCommonModule/thirdparty/prob_stats_segger/libpseggerswig.so
norm_so=${PyFilePath}/QueryRootCommonModule/thirdparty/normalication/_NomSwig.so,${PyFilePath}/QueryRootCommonModule/thirdparty/normalication/libnomalication_interface.so,${PyFilePath}/QueryRootCommonModule/thirdparty/normalication/libjson_linux-gcc-4.8.5_libmt.so
norm_file=${PyFilePath}/QueryRootCommonModule/sp_funs/dict/nomalication/rmap,${PyFilePath}/QueryRootCommonModule/sp_funs/dict/nomalication/special_prefix,${PyFilePath}/QueryRootCommonModule/sp_funs/dict/nomalication/tsmap
SUGGEST_DBPATH=hdfs://ns1013/user/recsys/suggest/app.db

echo "=====ok====="
$SPARK_HOME/bin/spark-submit \
--deploy-mode cluster \
--driver-memory 10G \
--executor-memory 20G \
--executor-cores 8 \
--conf spark.yarn.appMasterEnv.SUGGEST_DBPATH=hdfs://ns1013/user/recsys/suggest/app.db \
--conf spark.yarn.maxAppAttempts=2 \
--conf spark.shuffle.service.enabled=true \
--conf spark.dynamicAllocation.enabled=true \
--conf spark.dynamicAllocation.minExecutors=5 \
--conf spark.dynamicAllocation.maxExecutors=200 \
--conf spark.executor.memoryOverhead=16G \
--conf spark.broadcast.compress=true \
--conf spark.rdd.compress=true \
--conf spark.sql.hive.mergeFiles=true \
--conf spark.speculation=false \
--conf spark.shuffle.file.buffer=128k \
--conf spark.sql.broadcastTimeout=1000s \
--conf spark.reducer.maxSizeInFlight=96M \
--conf spark.shuffle.io.maxRetries=9 \
--conf spark.shuffle.io.retryWait=60s \
--conf spark.driver.extraLibraryPath=/software/servers/hadoop-2.7.1/lib/native \
--conf spark.executor.extraLibraryPath=/software/servers/hadoop-2.7.1/lib/native:/software/servers/hadoop-2.7.1/share/hadoop/common/lib/hadoop-lzo-0.4.20.jar \
--conf spark.pyspark.python=python2.7 \
--files $HIVE_CONF_DIR/hive-site.xml,${SUGGEST_DBPATH}/qp_common_file/correct_pair_black,${SUGGEST_DBPATH}/qp_common_file/correct_pair_white,${SUGGEST_DBPATH}/qp_common_file/mashup.dat.v1.4,${SUGGEST_DBPATH}/qp_common_file/mashup.dat.v2.0,${norm_file} \
--jars ${SUGGEST_DBPATH}/qp_common_file/jar/tensorflow-hadoop-1.11.0-rc2.jar,${SUGGEST_DBPATH}/xiala_common_file/jar/graphtest-mlrank-jar-with-dependencies.jar \
--conf spark.yarn.appMasterEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
--conf spark.executorEnv.yarn.nodemanager.container-executor.class=DockerLinuxContainer \
--conf spark.yarn.appMasterEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.jd.com:5000/wise_algorithm:latest \
--conf spark.executorEnv.yarn.nodemanager.docker-container-executor.image-name=bdp-docker.jd.com:5000/wise_algorithm:latest \
--py-files ${PyFilePath}/QueryRootCommonModule.zip,${PyFilePath}/settings.zip,${PyFilePath}/QueryRootCommonModule/thirdparty/lib/esm.so,${seg_so},${norm_so},${workspace}/src/qp_apps/session_adl/event_param_parser.py,${workspace}/src/qp_apps/session_adl/event_param_setting.py,${workspace}/src/qp_apps/applications/muti_task/conf.py \
$@
