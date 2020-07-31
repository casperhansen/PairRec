import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import glob
import argparse
from sklearn.neighbors import NearestNeighbors
import scipy

from nn_helpers import generator
from model import PairRecHash
import time
from gensim.models import KeyedVectors

def matrix_for_embed(w2v, mappings, maxId, size):
    '''
    Fill out the word embedding (w2v) based on the used mapping
    :param w2v: word embedding
    :param mappings: mapping from index to word (provided by a tokenizer)
    :param maxId: maximum index in mapping
    :param size: embedding size (given by the pretrained embedding, usually 2-300)
    '''
    matrix = np.random.uniform(-0.05, 0.05, (maxId, size))  # np.zeros((maxId,size[0]), dtype = np.float32)
    cc = 0
    for wordid in mappings:
        word = mappings[wordid]
        if word in w2v:
            matrix[wordid, :] = w2v[word]
            cc += 1
            #print("matrix_for_embed",word, wordid)
    print("### words found", cc/len(mappings))
    return matrix.astype(np.float32)

def make_word_embedding(vocab_size, embedding_dim, name="word_embedding", trainable=True, init=False):
    W = tf.Variable(tf.random_uniform(shape=[vocab_size, embedding_dim], minval=-0.05, maxval=0.05),
                    trainable=trainable, name=name)
    if init:
        embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
        embedding_init = W.assign(embedding_placeholder)
        return (W, embedding_placeholder, embedding_init)


def get_labels_and_indices(dname):
    collection = pickle.load(open("../data/" + dname + "_collections", "rb"))
    _, training, _, validation, testing, _, data_text_vect, labels, _, id2token = collection

    train_indices = training[-1]
    val_indices = validation[-1]
    test_indices = testing[-1]

    return labels, train_indices, val_indices, test_indices, data_text_vect, id2token

def eval_hashing(train_vectors, train_labels, val_vectors, val_labels, medianTrick=False):

    train_vectors = np.array(train_vectors)
    val_vectors = np.array(val_vectors)

    if medianTrick:
        medians = np.median(train_vectors, 0)
        train_vectors = (train_vectors > medians).astype(int)
        val_vectors = (val_vectors > medians).astype(int)

    upto = 100
    top100_precisions = []

    #for vali in range(len(val_vectors)):
    knn = NearestNeighbors(n_neighbors=upto, metric="hamming", n_jobs=-1)#, algorithm="brute")
    used_train_vectors = train_vectors
    use_val_vector = val_vectors
    use_val_labels = val_labels

    #used_train_vectors = used_train_vectors * use_val_vector[0]
    knn.fit(used_train_vectors)

    nns = knn.kneighbors(use_val_vector, upto, return_distance=False)
    for i, nn_indices in enumerate(nns):
        eval_label = use_val_labels[i]
        matches = np.zeros(upto)
        for j, idx in enumerate(nn_indices):
            if any([label in train_labels[idx] for label in eval_label]):
                matches[j] = 1
        top100_precisions.append(np.mean(matches))

    return top100_precisions

def extract_vectors_labels(sess, handle, specific_handle, num_samples, batch_placeholder, is_training, sigma_anneal_vae,
                           loss, emb_update, eval_batchsize, indices, labels, model, loss_doc_only):
    total = num_samples
    done = False
    losses = []
    losses_doc = []
    start = time.time()
    while not done:
        lossvals, _, loss_doc_only_vals = sess.run([loss, emb_update, loss_doc_only], feed_dict={handle: specific_handle, batch_placeholder: min(total, eval_batchsize),
                                                is_training: False, sigma_anneal_vae: 0})
        losses += lossvals.tolist()
        losses_doc += loss_doc_only_vals.tolist()

        total -= len(lossvals)
        if total <= 0:
            done = True

    #print("time", time.time() - start)
    losses = np.mean(losses)
    losses_doc = np.mean(losses_doc)
    embedding = sess.run(model.get_hashcodes())

    extracted_hashcodes = embedding[indices]
    print("ones:",np.sum(extracted_hashcodes)/np.sum(extracted_hashcodes>-1))
    extracted_labels = [labels[i] for i in indices]

    return losses, extracted_hashcodes, extracted_labels, losses_doc

def main():
    parser = argparse.ArgumentParser()
    # these are the most important to change
    parser.add_argument("--bits", default=8, type=int)
    parser.add_argument("--numnbr", default=25, type=int) # number of pairs from top k (default: k = 25)
    parser.add_argument("--dname", default="reuters", type=str)

    # the following values produce fine results, but should be tuned for obtaining the best result.
    parser.add_argument("--layersize", default=1000, type=int)
    parser.add_argument("--layers", default=2, type=int)
    parser.add_argument("--lr", default=0.001, type=float)

    # you can keep these fixed
    parser.add_argument("--eval_every", default=20000, type=int) # 20000
    parser.add_argument("--batchsize", default=40, type=int)
    parser.add_argument("--doc2weight", default=1.0, type=float)
    parser.add_argument("--hamweight", default=1.0, type=float)
    parser.add_argument("--KLweight", default=0.00, type=float)

    parser.add_argument("--dropout", default=0.8, type=float)
    parser.add_argument("--pretrainedwords", default=0, type=int)
    parser.add_argument("--max_seq_size", default=512, type=int)
    parser.add_argument("--maxham", default=-5, type=int)
    parser.add_argument("--hamweighting", default=0, type=int) # 0 = no, 1 = 1/(h+1), 2 = 1 - (H/bits)
    parser.add_argument("--maxiter", default=700000, type=int) # 0 = no, 1 = 1/(h+1), 2 = 1 - (H/bits)
    args = parser.parse_args()

    print(args)
    eval_batchsize = args.batchsize
    savename = "resultsNew/" + "_".join([str(v) for v in [args.dname, args.bits, args.batchsize, args.lr, args.KLweight, args.doc2weight,
                                                       args.maxham, args.numnbr, args.eval_every, args.hamweighting, args.dropout, args.layers, args.layersize, args.maxiter, args.pretrainedwords]])
    args = vars(args)


    if args["maxham"] > -0.5:
        maxham = args["maxham"]
        basepath = lambda v1, v2 : "../data/datasets/" + args["dname"] + "/maxham" + str(maxham) + "/" + str(v1) + "_maxham_sparse_" + str(v2)
    else: # just pick from the numnbr closest neighbours
        basepath = lambda v1, v2 : "../data/datasets/" + args["dname"] + "/sparse/" + str(v1) + "_sparse_" + str(v2)

    numnbr = args["numnbr"]
    trainfiles, valfiles, testfiles = [], [], []
    max_num_train_files = len(glob.glob("_".join(basepath("train",0).split("_")[:-1])+"*"))

    for i in range(min(max_num_train_files,numnbr)):
        trainfiles.append(basepath("train",i))
    valfiles.append(basepath("val",0))
    testfiles.append(basepath("test",0))

    print("\n\n\n#",trainfiles)
    #print(valfiles)
    #print(testfiles)
    #exit()
    #trainfiles = glob.glob(basepath + "*train_sparse_*")
    trainfiles_single = [trainfiles[0]]
    #valfiles = glob.glob(basepath + "*val_sparse_*")
    #testfiles = glob.glob(basepath + "*test_sparse_*")

    num_train_samples = sum(1 for _ in tf.python_io.tf_record_iterator(trainfiles[0])) * len(trainfiles)
    num_train_single_samples = int(num_train_samples/len(trainfiles))
    num_val_samples = sum(1 for _ in tf.python_io.tf_record_iterator(valfiles[0]))
    num_test_samples = sum(1 for _ in tf.python_io.tf_record_iterator(testfiles[0]))

    print(num_train_samples, num_train_single_samples, num_test_samples)

    labels, train_indices, val_indices, test_indices, data_text_vect, id2token = get_labels_and_indices(args["dname"])

    if args["pretrainedwords"]:
        w2v = KeyedVectors.load_word2vec_format(
            'gloveModelBin.bin',
            binary=True)
        embInit = matrix_for_embed(w2v, id2token, len(id2token), 300)

    num_dataset_samples = len(labels)
    bowlen = data_text_vect.shape[1]
    args["bowlen"] = bowlen

    print("----", bowlen)

    tf.reset_default_graph()
    with tf.Session() as sess:

        handle = tf.placeholder(tf.string, shape=[], name="handle_iterator")
        training_handle, train_iter, gen_iter = generator(sess, args["max_seq_size"], handle, args["batchsize"], trainfiles, 0, bowlen)
        train_single_handle, train_single_iter, _ = generator(sess, args["max_seq_size"], handle, eval_batchsize, trainfiles_single, 1, bowlen)
        val_handle, val_iter, _ = generator(sess, args["max_seq_size"], handle, eval_batchsize, valfiles, 1, bowlen)
        test_handle, test_iter, _ = generator(sess, args["max_seq_size"], handle, eval_batchsize, testfiles, 1, bowlen)

        sample = gen_iter.get_next()

        batch_placeholder = tf.placeholder(tf.int32, name="batch_placeholder") # use this to specify batchsize (for val/test it needs to be smaller in the last batch)
        is_training = tf.placeholder(tf.bool, name="is_training")
        sigma_anneal_vae = tf.placeholder(tf.float32, name="anneal_val", shape=())

        model = PairRecHash(sample, args, batch_placeholder, is_training, sigma_anneal_vae, num_dataset_samples)
        train_op, loss, emb_update, loss_doc_only, word_emb_matrix = model.make_network()

        if args["pretrainedwords"]:
            embedding_placeholder = tf.placeholder(tf.float32, [len(id2token), 300])
            embedding_init = word_emb_matrix.assign(embedding_placeholder)

        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(train_iter.initializer)

        if args["pretrainedwords"]:
            print("use pretrained!!!")
            sess.run(embedding_init, feed_dict={embedding_placeholder: embInit})

        running = True
        losses = []
        train_count = 0
        vae_val = 1.0
        vae_val_reduction = 1e-6

        patience_max = 10
        patience_current = 0
        best_val_loss = 100000000
        test_perf = []
        val_perf = []
        best_embeddings = None
        start_time = time.time()

        val_losses_list = []
        val_losses_doc_list = []

        while running:
            _, _, lossval = sess.run([train_op, emb_update, loss], feed_dict={handle: training_handle,
                                                                              batch_placeholder: args["batchsize"],
                                                                              is_training: True, sigma_anneal_vae: vae_val})
            losses += lossval.tolist()
            train_count += 1
            vae_val = max(vae_val - vae_val_reduction, 0)

            if train_count > 0 and train_count % args["eval_every"] == 0:
                print("Training", np.mean(losses), "vae_val", vae_val, "epochs", train_count*args["batchsize"]/num_train_single_samples)
                losses = losses[-(num_train_single_samples):]

                sess.run([train_single_iter.initializer,val_iter.initializer,test_iter.initializer])

                #allcodes = sess.run(model.get_hashcodes())
                #train_hashcodes = allcodes[train_indices]
                #train_labels = [labels[i] for i in train_indices]

                trainloss, train_hashcodes, train_labels, train_losses_doc = extract_vectors_labels(sess, handle, train_single_handle,
                                                                                  num_train_single_samples,
                                                                                  batch_placeholder, is_training,
                                                                                  sigma_anneal_vae, loss, emb_update,
                                                                                  eval_batchsize, train_indices, labels,
                                                                                  model, loss_doc_only)

                valloss, val_hashcodes, val_labels, val_losses_doc = extract_vectors_labels(sess, handle, val_handle,
                                                                                  num_val_samples,
                                                                                  batch_placeholder, is_training,
                                                                                  sigma_anneal_vae, loss, emb_update,
                                                                                  eval_batchsize, val_indices, labels,
                                                                                  model, loss_doc_only)

                testloss, test_hashcodes, test_labels, test_losses_doc = extract_vectors_labels(sess, handle, test_handle,
                                                                                  num_test_samples,
                                                                                  batch_placeholder, is_training,
                                                                                  sigma_anneal_vae, loss, emb_update,
                                                                                  eval_batchsize, test_indices, labels,
                                                                                  model, loss_doc_only)

                # train_hashcodes = data_text_vect[train_indices].todense()
                # train_labels = [labels[i] for i in train_indices]
                #
                # val_hashcodes = data_text_vect[val_indices].todense()
                # val_labels = [labels[i] for i in val_indices]
                #
                # test_hashcodes = data_text_vect[test_indices].todense()
                # test_labels = [labels[i] for i in test_indices]

                val_prec100 = eval_hashing(train_hashcodes, train_labels, val_hashcodes, val_labels)
                test_prec100 = eval_hashing(train_hashcodes, train_labels, test_hashcodes, test_labels)

                # print("Training", trainloss)
                # print("Validation", valloss, val_prec100)
                print("Testing", testloss, np.mean(test_prec100))

                if best_val_loss > valloss:
                    emb_matrix = sess.run(model.get_hashcodes())
                    best_embeddings = emb_matrix
                    best_val_loss = valloss
                    patience_current = 0
                    test_perf.append(test_prec100)
                    val_perf.append(np.mean(val_prec100))

                    val_losses_list.append(valloss)
                    val_losses_doc_list.append(val_losses_doc)

                    pickle.dump([best_embeddings, args, best_val_loss, train_count, vae_val, test_perf, val_perf,
                                 val_losses_list, val_losses_doc_list],
                                open(savename, "wb"))
                else:
                    patience_current += 1

                if patience_current >= patience_max or train_count > args["maxiter"]: #or ((time.time() - start_time) > (60*60*args["hours"])):
                    running = False

                if train_count < 400000:
                    patience_current = 0



if __name__ == '__main__':
    main()