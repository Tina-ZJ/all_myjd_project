# -*-encoding: utf-8 -*-

import argparse
import datetime
import os
import sys
import time
import random
import math

reload(sys)
sys.setdefaultencoding('utf8')

import tensorflow as tf
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import HiveContext, Row
from pyspark.sql.types import *
import copy
import json
import re
# from Norm import Norm
# from ProbSegger import ProbSegger

SUGGEST_DBPATH = os.environ['SUGGEST_DBPATH']
table_name = "app_algo_session_cids_bert_v_tfrecord"
index_table = "app_algo_session_cidx_idx"
output="hdfs://ns1013/user/recsys/suggest/app.db/qp_common_file/qp_personlization/"

def create_table():
    create_tb_sql = """
        CREATE EXTERNAL TABLE IF NOT EXISTS app.{table_name} (
            pvid                            string,
            keyword                         string,
            segs                            array<string>,
            input_ids                       array<int>,
            input_mask                      array<int>,
            segment_ids                      array<int>,
            cidx_list                       array<string>,
            cidx_id_list                         array<int>,
            cidx_name_list                       array<string>)
        PARTITIONED BY (dt string, version string)
        STORED AS ORC
        LOCATION '{SUGGEST_DBPATH}/{table_name}'
        TBLPROPERTIES('orc.compress'='SNAPPY')
    """.format(SUGGEST_DBPATH=SUGGEST_DBPATH, table_name=table_name)
    print("create_tb_sql:%s" % create_tb_sql)
    hiveCtx.sql(create_tb_sql)
    

    create_tb_sql = """
        CREATE EXTERNAL TABLE IF NOT EXISTS app.{index_table} (
            idx                            int,
            cidx                             string)
        PARTITIONED BY (dt string, dim_type string)
        STORED AS ORC
        LOCATION '{SUGGEST_DBPATH}/{index_table}'
        TBLPROPERTIES('orc.compress'='SNAPPY')
    """.format(SUGGEST_DBPATH=SUGGEST_DBPATH, index_table=index_table)
    print("create_idx_tb_sql:%s" % create_tb_sql)
    hiveCtx.sql(create_tb_sql)


def map_par_fun_seg(rows):
    ########## 归一化 #########
    from QueryRootCommonModule.thirdparty.normalication.Norm import Norm
    from QueryRootCommonModule.thirdparty.prob_stats_segger.ProbSegger import ProbSegger
    norm = Norm()
    norm.init()

    ############# 分词 #############
    psegger = ProbSegger()
    ret = psegger.Init('mashup.dat.v2.0')

    for r in rows:
        keyword = norm.process(r.keyword.encode('utf-8')).decode('utf-8')
        segs = psegger.Seg2(keyword)
        yield Row(pvid=r.pvid, keyword=keyword, segs=segs, gender_realt=r.gender_realt, cids_realt=r.cids_realt, brands_realt=r.brands_realt, age_realt=r.age_realt, search_keyword_seg_id=r.search_keyword_seg_id, cidx_list=r.cidx_list, cidx_name_list=r.cidx_name_list)


def run(sc, hiveCtx, dt_str, version):
    rdd = hiveCtx.sql("""
            select pvid, keyword, gender_realt, cids_realt, brands_realt, age_realt, search_keyword_seg_id, collect_set(cidx) as cidx_list, collect_set(cidx_name) as cidx_name_list from app.app_personalization_cid_samples
            where dt='{dt}' group by pvid, keyword, gender_realt, cids_realt, brands_realt, age_realt, search_keyword_seg_id """.format(dt=dt_str)).rdd.mapPartitions(map_par_fun_seg)
    
    rdd.cache()

            
    def flat_fun_cid(r):
        for x in r.cids_realt:
            yield x
    
    def flat_fun_brand(r):
        for x in r.brands_realt:
            yield x

    def to_id(terms, genders, cids_realt, brands_realt, ages, sessions, _id_dict, max_v, padding=40):    #35   33 28

        # query part
        term_id_arr = [_id_dict[t] if t in _id_dict else 1 for t in terms]
        query_input_ids = term_id_arr[:8]  #6
        for i in range(0, 8 - len(query_input_ids)):
            query_input_ids.append(0)

        query_input_mask = [ 1 if x>0 else 0 for x in query_input_ids]

        # user part
        genders = [_id_dict.get('__gender__'+str(x),1) for x in genders]
        ages = [_id_dict.get('__age__'+str(x),1) for x in ages]
        cids_realt = [_id_dict.get('__cid__'+str(x),1) if x>0 else 0 for x in cids_realt] 
        brands_realt = [_id_dict.get('__brand__'+str(x),1) if x>0 else 0 for x in brands_realt]
        user_input_ids = genders + ages + cids_realt + brands_realt
        user_input_mask = [ 1 if x>0 else 0 for x in user_input_ids]

        # session part
        session_input_ids = sessions
        session_input_mask = [ 1 if x>0 else 0 for x in sessions]
        
        # combine
        input_ids = user_input_ids + session_input_ids + query_input_ids
        input_mask = user_input_mask + session_input_mask + query_input_mask
        segment_ids = [0]*len(input_ids)
        
        if len(input_ids) == padding  and len(input_mask) == padding and len(segment_ids) == padding:
            return (input_ids, input_mask, segment_ids)
        else:
            None

    def cut(session,padding=25):
        session_id = [x for x in session]
        session_id = session_id[:padding]
        return session_id

    def cidx_to_id(cidx, cidx_id, padding=10):
        cidx_id_arr = [cidx_id[c] if c in cidx_id else -1 for c in cidx]
        if cidx_id_arr == len(cidx_id_arr) * [-1]:
            return None
        cidx_id_arr = cidx_id_arr[:padding]
        for i in range(0, padding - len(cidx_id_arr)):
            cidx_id_arr.append(-1)
        if len(cidx_id_arr) == padding:
            return cidx_id_arr
        else:
            None
   

    def sample(cidx, cidx_other_list, n):
        count = 0
        c = random.choice(cidx_other_list)
        while c == cidx and count<=n:
            c = random.choice(cidx_other_list)
            count+=1
        return c
    
    def map_fun_toid(iters):
        term_id_dict = br_term_id_dict.value
        cidx_id_dict = br_cidx_id_dict.value
        max_v = max(term_id_dict.values())
        for itr in iters:
            input_ids, input_mask, segment_ids = to_id(itr.segs, itr.gender_realt, itr.cids_realt, itr.brands_realt, itr.age_realt, itr.search_keyword_seg_id[:20], term_id_dict, max_v)
            cidids = cidx_to_id(itr.cidx_list, cidx_id_dict)
            #search_keyword_seg_id = cut(itr.search_keyword_seg_id)
            if cidids is not None:
                yield (itr.pvid, itr.keyword, itr.segs, input_ids, input_mask, segment_ids, itr.cidx_list, cidids, itr.cidx_name_list)

    term_id_dict = {}
    cidx_id_dict = {}

    
    realt_cidlist = rdd.flatMap(flat_fun_cid).distinct().collect()
    realt_brandlist = rdd.flatMap(flat_fun_brand).distinct().collect()
    cid_sql = """
            select cidx 
            from
            (
            select cidx, count(1) as sum from app.app_personalization_cid_samples where dt='{dt}' group by cidx
            )a
            join
            (
            select cate_id from dim.dim_item_category where valid_flag='1' and (cate_lvl_cd=3 or cate_lvl_cd=4)
            )b
            on a.cidx=b.cate_id
            where sum>20
            """.format(dt=dt_str)
            #100 50
    cidlist = hiveCtx.sql(cid_sql).rdd.collect()
    # get cidx index
    for i, c in enumerate(cidlist):
        cidx_id_dict.setdefault(c[0], i)
   
    # get valid cidx
    #cid_sql = """
    #        select cate_id, cate_name, lvl_2_cate_id, lvl_1_cate_id from dim.dim_item_category where valid_flag='1' and (cate_lvl_cd=3 or cate_lvl_cd=4)
    #        """

    # get term index
    term_sql = """
            select id, name from app.app_algo_emb_dict where version='union_v2' and dt='active' and type='term'
            """
    id_term = hiveCtx.sql(term_sql).rdd.collect()
    for x in id_term:
        term_id_dict.setdefault(x[1],int(x[0]))
    
    cid_len = len(realt_cidlist)
    max_v = max(term_id_dict.values())
    # add [cls] [sep]
    term_id_dict.setdefault('[cls]',max_v+1)
    term_id_dict.setdefault('[sep]',max_v+2)

    # add gender 4
    for i in range(4):
        x = '__gender__'+str(i)
        term_id_dict.setdefault(x,max_v+2+i+1)

    # add age 6
    for i in range(6):
        x = '__age__'+str(i)
        term_id_dict.setdefault(x,max_v+2+4+i+1)

    # add cid term
    for i, x in enumerate(realt_cidlist):
        x = '__cid__'+str(x)
        term_id_dict.setdefault(x, max_v+12+i+1)
    # add brand term
    for i, x in enumerate(realt_brandlist):
        x = '__brand__'+str(x)
        term_id_dict.setdefault(x, max_v+12+cid_len+i+1)

    # add [CLS] [SEP] 4个性别，6个年龄
    #max_sql = """
    #            select max(id) from app.app_algo_emb_dict where version='union' and dt='active' and type='term'
    #        """
    #max_id = hiveCtx.sql(max_sql).rdd.collect()



    
    sc.parallelize(cidx_id_dict.items()).map(lambda x:(x[1],x[0])).toDF(['idx', 'cidx']).registerTempTable("cidx_idx")
    sc.parallelize(term_id_dict.items()).map(lambda x:(x[1],x[0])).toDF(['idx', 'term']).registerTempTable("term_idx")

    # insert table
    hiveCtx.sql("""
        insert overwrite table app.{index_table} partition(dt='{dt}', dim_type='{dim_type}')
        select idx, cidx
        from cidx_idx
    """.format(index_table=index_table, dt=dt_str, dim_type=version+'_cidx'))

    hiveCtx.sql("""
        insert overwrite table app.{index_table} partition(dt='{dt}', dim_type='{dim_type}')
        select idx, term
        from term_idx
    """.format(index_table=index_table, dt=dt_str, dim_type=version+'_term'))
    # broadcast

    br_cidx_id_dict = sc.broadcast(cidx_id_dict)
    br_term_id_dict = sc.broadcast(term_id_dict)

    rdd.mapPartitions(map_fun_toid).toDF(['pvid','keyword', 'segs','input_ids','input_mask','segment_ids','cidx_list','cidx_id_list','cidx_name_list']).registerTempTable("tmp_table")
    hiveCtx.sql("""
        insert overwrite table app.{table_name} partition(dt='{dt}', version='{version}')
        select * from tmp_table
     """.format(table_name=table_name, dt=dt_str, version=version))



def toTFExample(x):
    input_ids_list = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(frame_) for frame_ in x.input_ids]))
    input_mask_list = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(frame_) for frame_ in x.input_mask]))
    segment_ids_list = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(frame_) for frame_ in x.segment_ids]))
    cidx_id_list = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(frame_) for frame_ in x.cidx_id_list]))
    seq_example = tf.train.Example(
        features=tf.train.Features(feature={
            "input_ids": input_ids_list,
            "input_mask": input_mask_list,
            "segment_ids": segment_ids_list,
            "cidx": cidx_id_list,
            }),
    )

    return seq_example.SerializeToString()
    
def run_tfrecord(sc, hiveCtx, dt_str, version):
    def binary_func(iters):
        for itr in iters:
            yield (bytearray(toTFExample(itr)), None)

    
    hiveCtx.sql("""
            select * from app.{table_name}
            where dt='{dt}' and version='{version}' and input_ids is not null order by rand()
            """.format(table_name=table_name, dt=dt_str, version=version)).rdd.mapPartitions(binary_func).\
                saveAsNewAPIHadoopFile(output+dt_str+'/'+'bert/'+version, "org.tensorflow.hadoop.io.TFRecordFileOutputFormat",
                                            keyClass="org.apache.hadoop.io.BytesWritable",
                                            valueClass="org.apache.hadoop.io.NullWritable")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--date", help="work date")
    parser.add_argument("-v", "--input_version", help="input version", default="v7")

    args = parser.parse_args()
    print("%s parameters:%s" % (sys.argv[0], args))

    begin_time = time.time()
    print("%s begin at %s" % (sys.argv[0], str(datetime.datetime.now())))

    conf = SparkConf()
    conf.set('spark.driver.cores', '4')
    conf.set('spark.driver.memory', '10g')
    conf.set('spark.executor.cores', '4')
    conf.set('spark.executor.memory', '20g')
    conf.set('spark.executor.memoryOverhead', '3g')
    conf.set('spark.sql.shuffle.partitions', '1000')
    conf.set('spark.default.parallelism', '1500')
    conf.set('spark.shuffle.service.enabled', 'true')
    conf.set('spark.dynamicAllocation.enabled', 'true')
    conf.set('spark.dynamicAllocation.minExecutors', '40')
    conf.set('spark.dynamicAllocation.maxExecutors', '200')
    conf.set('spark.shuffle.consolidateFiles', 'true')
    # conf.set("spark.sql.crossJoin.enabled", "true")
    sc = SparkContext(conf=conf, appName=(os.path.basename(sys.argv[0]) + args.date))
    sc.setLogLevel("WARN")
    hiveCtx = HiveContext(sc)
    hiveCtx.setConf('spark.shuffle.consolidateFiles', 'true')
    hiveCtx.setConf('spark.shuffle.memoryFraction', '0.4')
    hiveCtx.setConf('spark.sql.shuffle.partitions', '1000')
    dt = datetime.datetime.strptime(args.date, "%Y%m%d").date()
    dt_str = dt.strftime("%Y-%m-%d")
    hiveCtx.sql('use app')
    create_table()
    run(sc, hiveCtx, dt_str, args.input_version)
    run_tfrecord(sc, hiveCtx, dt_str, args.input_version) 
    print("Job Done!")
    end_time = time.time()
    print("%s end at %s" % (sys.argv[0], str(datetime.datetime.now())))
    print("%s total cost time:%s" % (sys.argv[0], str(end_time - begin_time)))
