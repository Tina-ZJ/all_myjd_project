# -*-coding=utf-8
import sys
import math
import datetime
from pyspark.sql.types import Row
from pyspark import SparkConf, SparkContext, HiveContext
import os
#import numpy as np

source_table='sz_algo_cid3_words_statistics_features'
save_table='sz_algo_query_term_weights_label'


def create_table(hiveCtx):

    create_tbl = """
        CREATE EXTERNAL TABLE IF NOT EXISTS app.{table_name} (
            query                                 string                     COMMENT   'query',
            pv                                    int                        COMMENT   'query周pv',
            terms                                 array<string>              COMMENT   'query分词',
            cid3                                  string                     COMMENT   'cid3',
            cid3_name                             string                     COMMENT   'cid3 名称',
            term_features                         map<string, array<float>>  COMMENT   '每个term对应的ctf, confidence, brand, icf, igm特征',
            ctf_icf                               array<string>              COMMENT   'ctf*icf计算出的term label分值',
            ctf_igm                               array<string>              COMMENT   'ctf*igm计算出的term label分值',
            ctf_confidence_icf                    array<string>              COMMENT   'ctf_confidence*icf 计算出的term label分值',
            ctf_confidence_igm                    array<string>              COMMENT   'ctf_confidence*igm 计算出的term label分值',
            ctf_brand_icf                         array<string>              COMMENT   '(ctf+brand*w)*icf 计算出的term label分值',
            ctf_brand_igm                         array<string>              COMMENT   '(ctf+brand*w)*igm 计算出的term label分值',
            ctf_confidence_brand_icf              array<string>              COMMENT   '(ctf_confidence+brand*w)*icf 计算出的term label分值',
            ctf_confidence_brand_igm              array<string>              COMMENT   '(ctf_confidence+brand*w)*igm 计算出的term label分值'
            )
        PARTITIONED BY(dt string, dim_type string)
        STORED AS ORC
        LOCATION 'hdfs://ns1013/user/recsys/suggest/app.db/{table_name}'
        TBLPROPERTIES('orc.compress'='SNAPPY')
    """.format(table_name=save_table)

    print(create_tbl)
    hiveCtx.sql(create_tbl)
  


def normalization(score_list, term_list, terms):
    sums = sum(score_list)
    score_list = [x/sums for x in score_list]
    term_score = dict()
    for s, t in zip(score_list, term_list):
        term_score[t] = s
    terms_score = [t+':'+str(term_score.get(t, 0.00000001)) for t in terms]

    return terms_score


def score_func(rows):
    for row in rows:
        
        query, pv, terms, cid3, cid3_name, term_list, brand_list, ctf_list, confidence_list, icf_list, igm_list = row
        score_list = list()
        if len(term_list)!=len(brand_list)!=len(ctf_list)!=len(confidence_list)!=len(icf_list)!=len(igm_list):
            continue
        terms_f = dict()
        for term, brand, ctf, confidence, icf, igm in zip(term_list, brand_list, ctf_list, confidence_list, icf_list, igm_list):
            brand = float(brand)
            ctf_icf = ctf*icf
            ctf_igm = ctf*igm
            ctf_confi_icf = confidence*icf
            ctf_confi_igm = confidence*igm
            ctf_brand_icf = (ctf + brand*0.2)*icf
            ctf_brand_igm = (ctf + brand*0.02)*igm
            ctf_confi_brand_icf = (confidence + brand*0.2)*icf
            ctf_confi_brand_igm = (confidence + brand*0.01)*igm
            score_list+=[ctf_icf, ctf_igm, ctf_confi_icf, ctf_confi_igm, ctf_brand_icf, ctf_brand_igm, ctf_confi_brand_icf, ctf_confi_brand_igm]

            # save term features
            terms_f.setdefault(term, [ctf, confidence, brand, icf, igm])

        score_result = [query, pv, terms, cid3, cid3_name, terms_f]
        # get normalization score
        for i in range(8):
            score = [score_list[t] for t in range(i, len(score_list), 8)]
            score_nor = normalization(score, term_list, terms)
            score_result.append(score_nor)
        yield score_result


def get_sample(hiveCtx):

    
######################### get query cid3 label #############################

    sql = """
                select key_word_qp_norm as query, query as pv, key_word_qp_termlist as terms, key_word_qp_cidinfo[0].f0_cid as cid3, key_word_qp_cidinfo[0].f1_name as cid3_name
                from app.app_m90_search_query_da_qp
                where dt='{dt}' and dim_type='{dim_type}' and version='jd_batch_inc_v2_han_v2' and query>7 and key_word_qp_norm is not null and size(key_word_qp_cidinfo)>0
                group by key_word_qp_norm, query, key_word_qp_termlist, key_word_qp_cidinfo[0].f0_cid, key_word_qp_cidinfo[0].f1_name
        """.format(dt='2021-11-24', dim_type='all_last_7')

    print(sql)
    hiveCtx.sql(sql).registerTempTable("table_query")



    sql="""
        select query, pv, terms, a.cid3 as cid3, cid3_name, a.term as term, brand_flag, ctf, ctf_confidence, icf, igm
        from
        (
            select query, pv, terms, cid3, cid3_name, term 
            from table_query
            LATERAL VIEW explode(terms) t as term
        )a
        join
        (
            select cid3, term, brand_flag, ctf, ctf_confidence, icf, igm
            from {source_table}
            where dt='{dt}' and dim_type='{dim_type}'
        )b
        on a.cid3=b.cid3 and a.term=b.term
        """.format(source_table=source_table, dt=dt_str, dim_type='entropy_v2')
    
    print(sql)
    hiveCtx.sql(sql).registerTempTable("table_term")

    sql="""
        select query, pv, terms, cid3, cid3_name, collect_list(term) as term_list, collect_list(brand_flag) as brand_list, collect_list(ctf) as ctf_list, 
        collect_list(ctf_confidence) as confidence_list, collect_list(icf) as icf_list, collect_list(igm) as igm_list
        from table_term
        group by query, pv, terms, cid3, cid3_name
        """

    print(sql)
    rdd = hiveCtx.sql(sql).rdd.cache()
    rdd.repartition(500).mapPartitions(lambda rows: score_func(rows))\
        .toDF(['query','pv', 'terms','cid3','cid3_name','term_features','ctf_icf','ctf_igm','ctf_confidence_icf','ctf_confidence_igm','ctf_brand_icf','ctf_brand_igm','ctf_confidence_brand_icf','ctf_confidence_brand_igm']).registerTempTable("score_table")
    
    insert_sql = """
                insert overwrite table {save_table} partition(dt='{dt}', dim_type='{dim_type}')
                select * from score_table
                """.format(save_table=save_table, dt=dt_str, dim_type='entropy')

    print("insert_sql:\n" + insert_sql)
    hiveCtx.sql(insert_sql)
    









if __name__=='__main__':
    conf = SparkConf()
    sc = SparkContext(conf=conf, appName="term_weight_ndcg_score")
    sc.setLogLevel("WARN")
    hiveCtx = HiveContext(sc)
    hiveCtx.setConf('spark.shuffle.consolidateFiles', 'true')
    hiveCtx.setConf('spark.shuffle.memoryFraction', '0.4')
    hiveCtx.setConf('spark.sql.shuffle.partitions', '1000')
    if len(sys.argv) == 1:
        dt = datetime.datetime.now() + datetime.timedelta(-1)
    else:
        dt = datetime.datetime.strptime(sys.argv[1], "%Y%m%d").date()

    dt_str = dt.strftime("%Y-%m-%d")
    yest_dt=dt + datetime.timedelta(-1)
    yest_str=yest_dt.strftime("%Y-%m-%d")

    hiveCtx.sql("use app")
    create_table(hiveCtx)
    get_sample(hiveCtx)


