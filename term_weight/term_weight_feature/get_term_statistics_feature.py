# -*-coding=utf-8
import sys

import datetime
from pyspark.sql.types import Row
from pyspark import SparkConf, SparkContext, HiveContext
import os

table_name='sz_algo_cid3_words_statistics_features'
table_sample_name='sz_algo_cid3_sku_samples'

def create_table(hiveCtx):
    create_tbl = """
        CREATE EXTERNAL TABLE IF NOT EXISTS app.{table_name} (
            cid3              string        COMMENT   'cid3',
            cid3_name         string        COMMENT   'cid3名称',
            term              string        COMMENT   '词',
            brand_flag        int           COMMENT   '该类目下的这个词是否是品牌词',
            sku_nums          int           COMMENT   '该类目下sku数量',
            term_cid3_nums    int           COMMENT   'term包含的cid3数量',
            neg_term_sku_nums int           COMMENT   '其它类目下包含该term的sku数量',
            term_sku_nums     int           COMMENT   '该类目下包含该term的sku数量',
            ctp               float         COMMENT   '类目下词平均位置: avg(term position in sku)',
            term_freq         int           COMMENT   '该类目下该term词频',
            ctf               float         COMMENT   '类目下该词频率: term_freq/sku_nums',
            ctf_confidence    float         COMMENT   '类目下词频率与该词下的最大词频相对差异: ctf/max(ctf), [0,1]之间，判断当前类目下的ctf置信度',
            all_cid3_nums     int           COMMENT   'cid3总数量', 
            rf                float         COMMENT   'relevance frequency: log(2+term_sku_nums/max(1,neg_term_sku_nums))',
            icf               float         COMMENT   '倒类目词频: log(all_cid3_nums/term_cid3_nums)',
            icf_prob          float         COMMENT   '倒类目词频prob: log((all_cid3_nums-term_cid3_nums)/term_cid3_nums)',
            icf_max           float         COMMENT   '倒类目词频max:  log(1+max(cid3)/term_cid3_nums)',
            igm               float         COMMENT   '全局类目反重力矩: max(ctf)/sum(ctf*rank)',
            entropy           float         COMMENT   '全局类目信息熵: sum(-plogp)'
            )
        PARTITIONED BY(dt string, dim_type string)
        STORED AS ORC
        LOCATION 'hdfs://ns1013/user/recsys/suggest/app.db/{table_name}'
        TBLPROPERTIES('orc.compress'='SNAPPY')
    """.format(table_name=table_name)

    print(create_tbl)
    hiveCtx.sql(create_tbl)
   
    create_tbl = """
         CREATE EXTERNAL TABLE IF NOT EXISTS app.{table_sample_name} (
            item_sku_id            string,
            cid3                   string,
            cid3_name              string,
            sku_seg               array<string>,
            brand_code              string
            )
        PARTITIONED BY(dt string, dim_type string)
        STORED AS ORC
        LOCATION 'hdfs://ns1013/user/recsys/suggest/app.db/{table_sample_name}'
        TBLPROPERTIES('orc.compress'='SNAPPY')
        """.format(table_sample_name=table_sample_name)
    print(create_tbl)
    hiveCtx.sql(create_tbl)




def get_features(hiveCtx):

    def main_func(rows):
        cid3_nums_dict = br_cid3_nums_dict.value
        for row in rows:
            item_sku_id, cid3, cid3_name, sku_seg, brand_chinese_name, brand_english_name = row
            sku_nums = cid3_nums_dict.get(cid3, 1)
            # filter 
            if sku_nums < 300:
                continue
            for i, seg in enumerate(sku_seg):
                brand_flag = 0
                if (seg==brand_chinese_name or seg==brand_english_name):
                    brand_flag = 1
                yield item_sku_id, cid3, cid3_name, seg, brand_flag, i, sku_nums

    
######################### get data ###########################################

    sql = """
        select  item_sku_id, item_last_cate_cd as cid3, item_last_cate_name as cid3_name, sku_name_qp_seg as sku_seg, brand_code
        from
        (
            select item_sku_id, item_last_cate_cd, item_last_cate_name, sku_name_qp_seg, brand_code, row_number() over(partition by item_last_cate_cd order by rand() ) as topn 
            from app.app_m03_search_item_sku_da_qp
            where dt='{dt}' and dim_type='{dim_type}'
        )
        where topn<=1000000
        """.format(dt=dt_str, dim_type=dim_type)

    print(sql)
    hiveCtx.sql(sql).registerTempTable("sku_seg")


######################## save original data ###################################

    insert_sql = """
                insert overwrite table {table_sample_name} partition(dt='{dt}', dim_type='{dim_type}')
                select  * from sku_seg
                """.format(table_sample_name=table_sample_name, dt=dt_str, dim_type=dim_type)
    print("insert_sql:\n" + insert_sql)
    hiveCtx.sql(insert_sql)
    
    

    

################################### get cid3 sku nums ###############


    sql = """
            select cid3, count(1) as sku_nums
            from {table_sample_name}
            where dt='{dt}' and dim_type='{dim_type}'
            group by cid3
        """.format(table_sample_name=table_sample_name, dt=dt_str, dim_type=dim_type)

    print(sql)
    cid3_nums = hiveCtx.sql(sql).rdd.collect()
    cid3_nums_dict = dict()
    for c, n in cid3_nums:
        cid3_nums_dict.setdefault(c, n)
    br_cid3_nums_dict = sc.broadcast(cid3_nums_dict)
   

##############################  brand_flag and ctp ##############################

    sql ="""
            select item_sku_id, cid3, cid3_name, sku_seg, goodbrand, brand_englistname
            from
            (
                select brandid, goodbrand, brand_englistname
                from app.goodbrand where dt='{dt}'  and brandid is not null
            )a
            right join
            (
                select item_sku_id, cid3, cid3_name, sku_seg, brand_code
                from {table_sample_name}
                where dt='{dt}' and dim_type='{dim_type}'
            )b
            on a.brandid=brand_code
        """.format(table_sample_name=table_sample_name, dt=dt_str, dim_type=dim_type)

    print(sql)
    data = hiveCtx.sql(sql).rdd.cache()
    data.mapPartitions(lambda rows: main_func(rows))\
    .toDF(['item_sku_id', 'cid3', 'cid3_name', 'term', 'brand_flag', 'position','sku_nums']).registerTempTable("sku_nums_infor")




######################################## ctf ##########################################################


    sql="""
        select cid3, cid3_name, b.term as term, brand_flag, sku_nums, term_cid3_nums, (term_sku_all_nums - term_sku_nums)/term_cid3_nums as neg_term_sku_nums, term_sku_nums, ctp, term_freq, term_freq/sku_nums as ctf
        from
        (
            select term, count(1) as term_sku_all_nums, size(collect_set(cid3)) as term_cid3_nums
            from sku_nums_infor
            group by term
        )a
        right join
        (
            select cid3, cid3_name, term, max(brand_flag) as brand_flag, sku_nums, count(1) as term_sku_nums, avg(position) as ctp, count(1) as term_freq
            from sku_nums_infor
            group by cid3, cid3_name, term, sku_nums
        )b
        on a.term=b.term
        """
    print(sql)
    hiveCtx.sql(sql).registerTempTable("term_ctf_temp")


################################# ctf confidence ################################################################

    sql="""
        select cid3, cid3_name, b.term as term, brand_flag, sku_nums, term_cid3_nums, neg_term_sku_nums, term_sku_nums, ctp, term_freq, ctf, ctf/ctf_max_ as ctf_confidence, ctf_max_ 
        from
        (
            select term, max(ctf) as ctf_max_
            from term_ctf_temp
            group by term

        )a
        right join
        (
            select cid3, cid3_name, term, brand_flag, sku_nums, term_cid3_nums, neg_term_sku_nums, term_sku_nums, ctp, term_freq, ctf
            from term_ctf_temp
        )b
        on a.term=b.term
        """
    print(sql)
    hiveCtx.sql(sql).registerTempTable("term_ctf")


###################################### icf ###################################################################
    
    sql="""
        select count(1) as num
        from
        (
            select cid3
            from term_ctf
            group by cid3
        )
        """
    print(sql)
    nums = len(cid3_nums_dict)
    sql="""
        select cid3, cid3_name, term,  brand_flag, sku_nums, term_cid3_nums, neg_term_sku_nums, term_sku_nums, ctp,  term_freq, ctf, ctf_confidence, ctf_max_, {nums} as all_cid3_nums , LOG({nums}/term_cid3_nums) as icf, LOG(({nums}-term_cid3_nums)/term_cid3_nums) as icf_prob
        from term_ctf
        """.format(nums=nums)
    print(sql)
    hiveCtx.sql(sql).registerTempTable("term_ctf_icf_temp")
         

################################### icf max ##########################################################

    sql="""
        select b.cid3 as cid3, cid3_name, term,  brand_flag, sku_nums, term_cid3_nums, neg_term_sku_nums, term_sku_nums, ctp, term_freq, ctf, ctf_confidence, ctf_max_, all_cid3_nums, icf, icf_prob, LOG(max_num/(1+term_cid3_nums)) as icf_max 
        from
        (
            select cid3, max(term_cid3_nums) as max_num
            from term_ctf_icf_temp
            group by cid3
        )a
        right join
        (
            select cid3, cid3_name, term,  brand_flag, sku_nums, term_cid3_nums, neg_term_sku_nums, term_sku_nums, ctp, term_freq, ctf, ctf_confidence, ctf_max_, all_cid3_nums, icf, icf_prob
            from term_ctf_icf_temp
        )b
        on a.cid3=b.cid3
        """
    print(sql)
    hiveCtx.sql(sql).registerTempTable("term_ctf_icf")



####################################  igm and entropy ###################################################################

    sql="""
            select term, sum(ctf*rank) as igm_d, max(ctf_max_)/sum(ctf*rank) as igm, sum(-ctf*LOG(ctf)) as entropy
            from
            (
            select term, ctf, ctf_max_, row_number() over(partition by term order by ctf desc) as rank
            from term_ctf_icf
            )
            group by term
        """
    print(sql)
    hiveCtx.sql(sql).registerTempTable("term_igm_entropy")


############################### rf and combine ################################################################

    sql="""
        select cid3, cid3_name, b.term as term,  brand_flag, sku_nums, term_cid3_nums, neg_term_sku_nums, term_sku_nums, ctp, term_freq, ctf, ctf_confidence, all_cid3_nums, LOG(2, 2+term_sku_nums/(1+neg_term_sku_nums)) as rf, icf, icf_prob, icf_max, igm, entropy
        from
        (
            select term, igm, entropy
            from term_igm_entropy
        )a
        right join
        (
            select cid3, cid3_name, term,  brand_flag, sku_nums, term_cid3_nums, neg_term_sku_nums, term_sku_nums, ctp, term_freq, ctf, ctf_confidence, all_cid3_nums, icf, icf_prob, icf_max
            from term_ctf_icf
        )b
        on a.term=b.term
        """
    print(sql)
    hiveCtx.sql(sql).registerTempTable("term_all_infor")



    insert_sql = """
                insert overwrite table {table_name} partition(dt='{dt}', dim_type='{dim_type}')
                select  * from term_all_infor
                """.format(table_name=table_name, dt=dt_str, dim_type=dim_type)
    print("insert_sql:\n" + insert_sql)
    hiveCtx.sql(insert_sql)
    
    


if __name__ == "__main__":
    conf = SparkConf()
    sc = SparkContext(conf=conf, appName="sz_algo_recallrelation_query_score_feature")
    sc.setLogLevel("WARN")
    hiveCtx = HiveContext(sc)
    hiveCtx.setConf('spark.shuffle.consolidateFiles', 'true')
    hiveCtx.setConf('spark.shuffle.memoryFraction', '0.4')
    hiveCtx.setConf('spark.sql.shuffle.partitions', '1000')
    if len(sys.argv) == 1:
        dt = datetime.datetime.now() + datetime.timedelta(-1)
    else:
        dt = datetime.datetime.strptime(sys.argv[1], "%Y%m%d").date()
    dim_type='base'
    dt_str = dt.strftime("%Y-%m-%d")
    yest_dt=dt + datetime.timedelta(-1)
    yest_str=yest_dt.strftime("%Y-%m-%d")

    hiveCtx.sql("use app")
    create_table(hiveCtx)
    get_features(hiveCtx)
