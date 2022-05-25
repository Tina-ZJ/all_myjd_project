hive -e "select query, terms, cid3, cid3_name, term_weights, term, features, label from app.sz_algo_query_terms_rank_samples limit 1000000 " >data.100w
