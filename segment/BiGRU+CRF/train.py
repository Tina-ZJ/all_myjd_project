# coding:utf-8
import time
import helper
import argparse
from BILSTM_CRF import BILSTM_CRF
from functools import wraps
from ModelUtil import *


def assign(sess, params):
    for var in tf.trainable_variables():
        if params.has_key(var.name):
            data2 = params[var.name]
            sess.run(tf.assign(var, data2))


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Total time running %s : %s h" % (function.func_name, str(((t1 - t0) / 3600))))
        return result

    return function_timer


def parser_paramter():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="the name of the train,eg: seg, pos, ner")
    parser.add_argument("train_path", help="the path of the train file")
    parser.add_argument("save_path", help="the path of the saved model")
    parser.add_argument("-fine_tune", "--fine_tune", help="whether fine_tune ", default=False)
    parser.add_argument("-vr", "--val_ratio", help="the validation rate, the default is 0.95", default=0.95, type=float)
    parser.add_argument("-e", "--epoch", help="the number of epoch", default=7, type=int)
    parser.add_argument("-ce", "--char_emb", help="the char embedding file", default='emb/query.word.300d.vec') #default ='emb/query.word.300d.vec'
    parser.add_argument("-xpu", "--xpu", help="the configure of gpu or cpu, the default is /gpu:0", default="/gpu:1",
                        type=str)
    parser.add_argument("-lr", "--learning_rate", help="the learning rate, the default is 0.001", default=0.001,
                        type=float)
    parser.add_argument("-bs", "--batch_size", help="the batch size, the default is 256", default=256, type=int)
    parser.add_argument("-m", "--emb_dim", help="the embedding dim, the default is 300", default=300, type=int)
    parser.add_argument("-hd", "--hidden_dim", help="the hidden layer dim, the default is 200", default=200, type=int)
    parser.add_argument("-nl", "--num_layers", help="the number of lstm layers, the default is 1", default=1, type=int)
    parser.add_argument("-ns", "--num_steps", help="the number of steps, the default is 200", default=80, type=int)
    parser.add_argument("-dr", "--dropout_rate", help="the learning rate, the default is 0.5", default=0.8, type=float)

    args = parser.parse_args()

    name = args.name
    train_path = args.train_path
    save_path = args.save_path
    fine_tune = args.fine_tune
    num_epochs = args.epoch
    emb_path = args.char_emb
    xpu_config = args.xpu
    learning_rate = args.learning_rate
    emb_dim = args.emb_dim
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    num_steps = args.num_steps
    dropout_rate = args.dropout_rate
    val_ratio = args.val_ratio
    batch_size = args.batch_size
    return (name, train_path, save_path, fine_tune, num_epochs, emb_path,
            xpu_config, learning_rate, emb_dim, hidden_dim, num_layers,
            num_steps, dropout_rate, val_ratio, batch_size)


@fn_timer
def bilstm_train():
    (name, train_path, save_path, fine_tune, num_epochs, emb_path,
     xpu_config, learning_rate, emb_dim, hidden_dim, num_layers,
     num_steps, dropout_rate, val_ratio, batch_size) = parser_paramter()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    char2id_file = os.path.join(save_path, "char2id")
    label2id_file = os.path.join(save_path, "label2id")

    print "preparing train and validation data"
    char2id, id2char, label2id, id2label = helper.buildMap(train_path, char2id_file, label2id_file)

    x_train, y_train, x_val, y_val = helper.getTrain(train_path, char2id, label2id, num_steps,
                                                     train_val_ratio=val_ratio)
    num_chars = len(id2char.keys())
    num_classes = len(id2label.keys())
    if emb_path is not None:
        embedding_matrix = helper.getEmbedding(emb_path, char2id_file)
        num_chars = embedding_matrix.shape[0]
        emb_dim = embedding_matrix.shape[1]
    else:
        embedding_matrix = None

    print "num_chars: " + str(num_chars) + ", emb_dim: " + str(emb_dim) + ", num_classes: " + str(num_classes)
    print "building model"

    # GPU memory restrict
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
        with tf.device(xpu_config):
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            with tf.variable_scope(name, reuse=None, initializer=initializer):
                model = BILSTM_CRF(num_chars=num_chars, num_classes=num_classes, num_steps=num_steps,
                                   emb_dim=emb_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                                   num_epochs=num_epochs, learning_rate=learning_rate,
                                   dropout_rate=dropout_rate, batch_size=batch_size,
                                   embedding_matrix=embedding_matrix)
            params = [("name", name),
                      ("max_seq_len", num_steps),
                      ("emb_dim", emb_dim),
                      ("hidden_dim", hidden_dim),
                      ("num_layers", num_layers),
                      ("dropout_rate", dropout_rate),
                      ("lstm", model.bLstm),
                      ("use_peepholes", model.use_peepholes)]
            helper.saveModelParameters(save_path, params)
            print "training model"
            tf.global_variables_initializer().run()

            # fine_tune
            if fine_tune:
                params = read_pkl()
                assign(sess, params)

            model.train(sess, os.path.join(save_path, "model"), x_train, y_train, x_val, y_val)


if __name__ == '__main__':
    bilstm_train()
