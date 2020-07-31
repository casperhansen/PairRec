import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops


class PairRecHash():
    def __init__(self, sample, args, batchsize, is_training, sigma_anneal_vae, num_dataset_samples):
        self.sample = sample
        self.args = args

        self.batchsize = batchsize
        self.is_training = is_training
        self.sigma_anneal_vae = sigma_anneal_vae

        self.num_dataset_samples = num_dataset_samples

    #################### Bernoulli Sample #####################
    ## ref code: https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
    def bernoulliSample(self, x):
        """
        Uses a tensor whose values are in [0,1] to sample a tensor with values in {0, 1},
        using the straight through estimator for the gradient.
        E.g.,:
        if x is 0.6, bernoulliSample(x) will be 1 with probability 0.6, and 0 otherwise,
        and the gradient will be pass-through (identity).
        """
        g = tf.get_default_graph()
        with ops.name_scope("BernoulliSample") as name:
            with g.gradient_override_map({"Ceil": "Identity", "Sub": "BernoulliSample_ST"}):
                train_fn = lambda: tf.minimum(tf.ones(tf.shape(x)), tf.random_uniform(tf.shape(x)))
                eval_fn = lambda: tf.minimum(tf.ones(tf.shape(x)), tf.ones(tf.shape(x)) * 0.5)

                mus = tf.cond(self.is_training, train_fn, eval_fn)
                return tf.ceil(x - mus, name=name)

    @ops.RegisterGradient("BernoulliSample_ST")
    def bernoulliSample_ST(op, grad):
        return [grad, tf.zeros(tf.shape(op.inputs[1]))]

    ###########################################################

    def encoder(self, docbow):

        doc_layer = tf.layers.dense(docbow, self.args["layersize"], name="encode_layer0", reuse=tf.AUTO_REUSE, activation=tf.nn.relu)

        for i in range(1, self.args["layers"]):
            doc_layer = tf.layers.dense(doc_layer, int(self.args["layersize"]/i), name="encode_layer" + str(i),reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
            #doc_layer = tf.nn.dropout(doc_layer, tf.cond(self.is_training, lambda: 0.8, lambda: 1.0))

        doc_layer = tf.nn.dropout(doc_layer, tf.cond(self.is_training, lambda: self.args["dropout"], lambda: 1.0))
        #doc_layer = tf.layers.batch_normalization(doc_layer, training=self.is_training)
        sampling_vector = tf.layers.dense(doc_layer, self.args["bits"], name="last_encode", reuse=tf.AUTO_REUSE, activation=tf.nn.sigmoid)


        bit_vector = self.bernoulliSample(sampling_vector)

        bit_vector_det = tf.ceil(sampling_vector - tf.ones(tf.shape(sampling_vector))*0.5)

        return bit_vector, bit_vector_det, sampling_vector

    def make_noisy_hashcode(self, hashcode):
        e = tf.random.normal([self.batchsize, self.args["bits"]])
        return tf.math.multiply(e, self.sigma_anneal_vae) + hashcode

    def compute_KL(self, sampling_vector):
        loss_kl = tf.multiply(sampling_vector, tf.math.log(tf.maximum(sampling_vector / 0.5, 1e-10))) + \
                  tf.multiply(1 - sampling_vector, tf.math.log(tf.maximum((1 - sampling_vector) / 0.5, 1e-10)))
        loss_kl = tf.reduce_sum(tf.reduce_sum(loss_kl, 1), axis=0)
        return loss_kl

    def decoder(self, hashcode, sampling_vector, target):
        noisy_hashcode = self.make_noisy_hashcode(hashcode)
        # decode_layer = tf.layers.dense(noisy_hashcode, target.shape[1], name="decode0", reuse=tf.AUTO_REUSE)
        kl_loss = self.compute_KL(sampling_vector)
        # sqr_diff = tf.math.pow(decode_layer - target, 2)
        # mse = tf.reduce_mean(sqr_diff, axis=-1)
        # loss = mse + self.args["KLweight"]*kl_loss

        #embedding = tf.layers.dense(self.word_emb_matrix, self.args["bits"], name="lower_dim_embedding_layer", reuse=tf.AUTO_REUSE)
        dot_emb_vector = tf.linalg.matmul(noisy_hashcode,tf.transpose(self.word_emb_matrix * tf.expand_dims(self.importance_emb_matrix,-1))) + self.softmax_bias
        softmaxed = tf.nn.softmax(dot_emb_vector)
        logaritmed = tf.math.log(tf.maximum(softmaxed, 1e-10))
        logaritmed = tf.multiply(logaritmed, tf.cast(target > 0, tf.float32))
        loss_recon = tf.reduce_sum(logaritmed, 1)
        loss = -(loss_recon - self.args["KLweight"]*kl_loss)
        return loss

    def get_hashcodes(self):
        return self.hashcode_embedding

    def make_network(self):
        emb_size = self.args["bits"]
        emb_dtype = tf.int8

        self.hashcode_embedding = tf.Variable(tf.zeros(shape=[self.num_dataset_samples, emb_size], dtype=emb_dtype), trainable=False, name="hashcode_emb")

        #doc_input_ids, doc_input_mask, doc_segment_ids, doc2_input_ids, doc2_input_mask, \
        #doc2_segment_ids, weak_hamming_dist, doc_idx, doc2_idx, doc_bow, doc2_bow = self.sample

        _, _, weak_hamming_dist, doc_idx, doc2_idx, doc_bow, doc2_bow = self.sample

        self.importance_emb_matrix = tf.Variable(tf.random_uniform(shape=[self.args["bowlen"]], minval=0.1, maxval=1),
                        trainable=True, name="importance_embedding")

        word_emb_matrix_big = tf.Variable(tf.random_uniform(shape=[self.args["bowlen"], 300], minval=-0.05, maxval=0.05),
                                                 trainable=True, name="word_embedding")
        if self.args["pretrainedwords"]:
            self.word_emb_matrix = tf.layers.dense(word_emb_matrix_big, self.args["bits"], name="word_embedding_bits")
        else:
            self.word_emb_matrix = tf.Variable(tf.random_uniform(shape=[self.args["bowlen"], self.args["bits"]], minval=-1, maxval=1),
                                                 trainable=True, name="word_embedding")

        doc_bow = doc_bow * self.importance_emb_matrix
        doc2_bow = doc2_bow * self.importance_emb_matrix

        # Encoding
        doc_hashcode, doc_det_hashcode, doc_sampling = self.encoder(doc_bow)
        doc2_hashcode, _, doc2_sampling = self.encoder(doc2_bow)

        # Decoding
        self.softmax_bias = tf.Variable(tf.zeros(self.args["bowlen"]), name="softmax_bias")
        doc_loss = self.decoder(doc_hashcode, doc_sampling, doc_bow)
        doc2_loss = self.decoder(doc2_hashcode, doc2_sampling, doc_bow)

        if self.args["hamweighting"] == 1:
            weight = (1.0/(tf.cast(weak_hamming_dist, tf.float32)+1))
            loss = doc_loss + self.args["doc2weight"] * doc2_loss * weight
        elif self.args["hamweighting"] == 2:
            weight = 1.0 - tf.cast(weak_hamming_dist, tf.float32)/self.args["bits"]
            loss = doc_loss + self.args["doc2weight"] * doc2_loss * weight
        elif self.args["hamweighting"] == 0:
            loss = doc_loss + self.args["doc2weight"]*doc2_loss
        else:
            print("unknown hamweighting value")
            exit(-1)

        doc_hashcode_h = 2*doc_hashcode - 1
        doc2_hashcode_h = 2*doc2_hashcode - 1
        fixed_weak_hamming_dist = self.args["bits"] - 2*tf.cast(weak_hamming_dist, tf.float32)
        dotprod = tf.reduce_sum(doc_hashcode_h*doc2_hashcode_h, -1)
        hamming_loss = self.args["hamweight"] * (dotprod - fixed_weak_hamming_dist)**2

        train_op = tf.train.AdamOptimizer(learning_rate=self.args["lr"],name="AdamOptimizer") #tf.train.AdamOptimizer(learning_rate=initial_learning_rate)
        train_op = train_op.minimize(loss)

        emb_update = tf.scatter_update(self.hashcode_embedding, doc_idx, tf.cast(doc_hashcode, emb_dtype))

        return train_op, loss, emb_update, doc_loss, word_emb_matrix_big