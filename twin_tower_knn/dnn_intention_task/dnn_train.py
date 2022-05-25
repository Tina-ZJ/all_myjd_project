# -*- coding: utf-8 -*-
import sys
import tensorflow as tf
import numpy as np
from dnn_model import DNN
from data_util import create_term
import os
import codecs
import batch_read_tfrecord
import common
import traceback
from read_file import run_shell_cmd
#import word2vec

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes",10,"number of label")
tf.app.flags.DEFINE_float("learning_rate",0.001,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 512, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("decay_steps", 800000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 0.95, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_string("ckpt_dir","checkpoint/DNN/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length",8,"max sentence length")
tf.app.flags.DEFINE_integer("cid_length",6,"max sentence length")
tf.app.flags.DEFINE_integer("product_length",6,"max sentence length")
tf.app.flags.DEFINE_integer("brand_length",6,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",200,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",4,"number of epochs to run.")
tf.app.flags.DEFINE_integer("batch_save", 5000, "save model how many batch")
tf.app.flags.DEFINE_boolean("use_embedding",False,"whether to use embedding or not.")
tf.app.flags.DEFINE_string("train_sample_file","data/train_sample.tfrecord","path of traning data.")
tf.app.flags.DEFINE_integer("hidden_size",128,"hidden size")
tf.app.flags.DEFINE_string("term_index_path","data/term_index.txt",'path of term')
tf.app.flags.DEFINE_string("summary_dir","./summary/",'path of summary')
tf.app.flags.DEFINE_integer("train_sample_num",273000000,"train sample num")

#command = "hadoop fs -ls hdfs://ns1013/user/recsys/suggest/app.db/qp_common_file/multi/query-sku-reverse-session-month-v3_tfrecord_n2/ | grep part | awk '{print $NF}' 2>/dev/null"
#input_files = run_shell_cmd(command)
train_sample_file = FLAGS.train_sample_file

def main(_):
    #1. load vocabulary
    vocabulary_word2index, vocabulary_index2word= create_term(FLAGS.term_index_path)
    vocab_size = len(vocabulary_index2word)
    print("vocab_size:",vocab_size)
    
    #2.create session.
    with tf.Graph().as_default():
        config=tf.ConfigProto()
        config.gpu_options.allow_growth=True
        model = DNN( FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate, vocab_size, FLAGS.embed_size, FLAGS.hidden_size, FLAGS.is_training)
        train_batcher = batch_read_tfrecord.SegBatcher(train_sample_file, FLAGS.batch_size,num_epochs=FLAGS.num_epochs)
        global_init_op = tf.global_variables_initializer()
        local_init_op = tf.local_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            sess.run(global_init_op)
            sess.run(local_init_op)
            if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
                print("Restoring Variables from Checkpoint")
                saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            else:
                print('Initializing Variables')
                if FLAGS.use_embedding:
                    assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, model)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            curr_epoch=sess.run(model.epoch_step)
            #3.feed data & training
            train_sample_num = FLAGS.train_sample_num

            for epoch in range(curr_epoch,FLAGS.num_epochs):
                train_example_num = 0
                counter = 0
                while train_example_num < train_sample_num:
                    try :
                        train_batch_data = sess.run(train_batcher.next_batch_op)
                        query, cid, cid_neg, product, product_neg, labels = train_batch_data
                        if len(query)!=FLAGS.batch_size:
                            continue
                        feed_dict = {model.query: query, model.cid: cid, model.product: product, model.cid_neg: cid_neg, model.product_neg: product_neg, model.intention_label: labels, model.dropout_keep_prob: 0.5}

                        global_step, loss, acc, _=sess.run([model.global_step, model.loss_val, model.accuracy, model.train_op],feed_dict)
                        counter+=1
                        train_example_num += FLAGS.batch_size

                        if not os.path.exists(FLAGS.ckpt_dir):
                            os.makedirs(FLAGS.ckpt_dir)
                        if counter % FLAGS.batch_save==0:
                            save_path=FLAGS.ckpt_dir+"model.ckpt"
                            saver.save(sess,save_path,global_step=epoch)
                            print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain acc:%.3f\t" %(epoch, counter, loss, acc))
                    except tf.errors.OutOfRangeError:
                        print("Done Training")
                        break

                #epoch increment
                print("going to increment epoch counter....")
                print(epoch)
                sess.run(model.epoch_increment)
                
                #save every epoch model to checkpoint
                save_path=FLAGS.ckpt_dir+"model.ckpt"
                saver.save(sess,save_path,global_step=epoch)
        coord.request_stop()
        coord.join(threads)
        sess.close()

def assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, model):
    print("using pre-trained word emebedding.started.word2vec_model_path:",word2vec_model_path)
    f = open(word2vec_model_path)
    word2vec_dict = {}
    for line in f:
        terms = line.strip().split()
        if len(terms)!=301:
            continue
        word2vec_dict[terms[0]] = np.array([float(x) for x in terms[1:]]) 
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0;
    count_not_exist = 0
    for i in range(1, vocab_size):  # loop each word
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding;
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(model.Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")


if __name__ == "__main__":
    tf.app.run()
