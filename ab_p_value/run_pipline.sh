source ./conf.sh

if [ $# -eq 2 ]
then
    dt_s="$1"
	dt_e="$2"
else
    echo "$0 dt_s dt_e"
	exit -1
fi


echo "dt:${dt_s}~${dt_e}"
echo "base:${base};test:${test}"
echo "dim:$dim"

mkdir -p logs

echo "Compute variance..."
sh run_spark_arg.sh app_search_ab_pvalue.py
if [ $? -ne 0 ]
then
    echo "Compute variance failed..."
    exit -1
fi

mkdir -p tmp
info_file="./tmp/info.${dt_e}.${dim}.dat"
hive -e "select version, pv, uv, click, uclick, orderlines, gmv, uv_value, ucvr, cvr, ctr from app.app_ab_mtest_base_szda where dt='${dt_e}' and dim_type='${dim}'" > $info_file
variance_file="./tmp/variance.${dt_e}.${dim}.dat"
hive -e "select indicator_type, base_variance, base_mean, base_simlpe_size, test_variance, test_mean, test_simlpe_size from app.app_ab_mtest_pvalue_szda where dt='${dt_e}' and dim_type='${dim}'" > $variance_file

echo "Compute p value..."
python compute_pvalue.py --file_i=$info_file --file_v=$variance_file
if [ $? -ne 0 ]
then
    echo "Compute p value failed..."
    exit -1
fi




