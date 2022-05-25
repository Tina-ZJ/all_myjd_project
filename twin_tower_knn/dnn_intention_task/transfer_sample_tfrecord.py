#-*-coding=utf-8-*-


import re
import codecs
import tensorflow as tf
import common
import argparse

def sample_to_tfrecord(sample_file, tfrecord_file, term_index):
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    num = 0
    with open(sample_file) as f:
        for i, line in enumerate(f):
            num += 1
            example = tf.train.SequenceExample()
            fl_terms = example.feature_lists.feature_list["terms"]
            fl_pos_cid = example.feature_lists.feature_list["pos_cid"]
            fl_neg_cid = example.feature_lists.feature_list["neg_cid"]
            fl_pos_pdt = example.feature_lists.feature_list["pos_pdt"]
            fl_neg_pdt = example.feature_lists.feature_list["neg_pdt"]
            fl_labels = example.feature_lists.feature_list["labels"]
            tokens = line.strip().split('\t')
            if tokens[0].strip()!='' and tokens[0].strip()!='NULL':
                terms = tokens[1].strip().split(',')
                pos_cid = tokens[2].strip().split(',')
                neg_cid = tokens[3].strip().split(',')
                pos_pdt = tokens[4].strip().split(',')
                neg_pdt = tokens[5].strip().split(',')
                label = tokens[-1].strip()
                while len(terms) <8:
                    terms.append('<PAD>')
                for x in [pos_cid, neg_cid]:
                    while len(x) <6:
                        x.append('<PAD>')
                for x in [pos_pdt, pos_pdt]:
                    while len(x) <2:
                        x.append('<PAD>')
                fl_labels.feature.add().int64_list.value.append(int(label))
                for term in terms:
                    if term in term_index:
                        fl_terms.feature.add().int64_list.value.append(term_index[term])
                    else:
                        fl_terms.feature.add().int64_list.value.append(1)
                for term in pos_cid:
                    if term in term_index:
                        fl_pos_cid.feature.add().int64_list.value.append(term_index[term])
                    else:
                        fl_pos_cid.feature.add().int64_list.value.append(1)
                for term in neg_cid:
                    if term in term_index:
                        fl_neg_cid.feature.add().int64_list.value.append(term_index[term])
                    else:
                        fl_neg_cid.feature.add().int64_list.value.append(1)
                for term in pos_pdt:
                    if term in term_index:
                        fl_pos_pdt.feature.add().int64_list.value.append(term_index[term])
                    else:
                        fl_pos_pdt.feature.add().int64_list.value.append(1)
                for term in neg_pdt:
                    if term in term_index:
                        fl_neg_pdt.feature.add().int64_list.value.append(term_index[term])
                    else:
                        fl_neg_pdt.feature.add().int64_list.value.append(1)
            writer.write(example.SerializeToString())
            if i % 1000000 == 0:
                print('{i} line sample transfer succeed....'.format(i=i))
                writer.flush()
    return num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_sample_file", nargs='?', default='./data/train_sample.txt')
    parser.add_argument("--term_index_file", nargs='?', default='./data/term_index.txt')

    args = parser.parse_args()
    train_tfrecord_file = args.train_sample_file[:-4] + '.tfrecord'
    _, term_index = common.get_term_index(args.term_index_file)
    train_sample_size = sample_to_tfrecord(args.train_sample_file, train_tfrecord_file, term_index)
    print('train sample transfer to tfrecord succeed....')
