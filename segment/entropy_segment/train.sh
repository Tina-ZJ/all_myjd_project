cat ../data.v1/all.same.rule2.JDSeg.uniq.combineBrand | python term.py | python infor.py left right 
python combine.py left right infor_all

sort -t $'\t' -k 2rn -k 3rn infor_all >infor_all_sort

cat infor_all_sort | python filter.py infor_all_sort_filter


对bad进行AC匹配， 得到 bad.JDTag, 然后把是产品和品牌的提取回来

python extract.py bad.JDTag good.JDTag

cat good.JDTag >> infor_all_sort_filter

对产品词表，和品牌词表，产品词表抽取为3个字的产品，匹配词表抽取大于等于三个字，小于等于5个字的，还有英语品牌，
extract_product extract_brand english.brand 全部左右信息熵复制为1 然后再与infor_all_sort_filter合并，得到

infor_all_sort_filter.extract_product.brand.english

cat infor_all_sort_filter.extract_product.brand.english | python augment.py >infor_all_sort_filter.extract_product.brand.englis.augment




最后，
cat infor_all_sort_filter.extract_product.brand.english_augment | python filter2.py > infor_all_sort_filter.extract_product.brand.english_augment_filter

cat ***** | python augment2.py   >*******


发现一个特别头痛的问题，弄了好久，java在建立树的时候，会对特殊符号去掉，在python的时候是不处理的，所以两个term是有区别的，但是要是java去掉特殊符号后
可能有相同的两个term，然后建立树的时候，相同的term会覆盖，所以在最后面的那个term的分值才是最终的，要么把代码那里预处理去掉，要么把最重要的term分值放后面
或者排个序，分值越大的在后面
