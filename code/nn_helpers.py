import tensorflow as tf




def generator(sess, maxsize, handle, batchsize, record_paths, is_test, bowlen):
    def extract_fn(data_record):
        '''
            features["doc_bow_data"] = create_float_feature(doc_bow_data)
            features["doc_bow_indices"] = create_int_feature(doc_bow_indices)
            features["doc2_bow_data"] = create_float_feature(doc2_bow_data)
            features["doc2_bow_indices"] = create_int_feature(doc2_bow_indices)
            features["doc_bert"] = create_float_feature(doc_bert)
            features["doc2_bert"] = create_float_feature(doc2_bert)

            features["weak_hamming_dist"] = create_int_feature([int(weak_hamming_dist)])
            features["doc_idx"] = create_int_feature([int(doc_idx)])
            features["doc2_idx"] = create_int_feature([int(doc2_idx)])

        '''

        features = {
            'doc_bow_data': tf.VarLenFeature(tf.float32),
            'doc_bow_indices': tf.VarLenFeature(tf.int64),
            'doc2_bow_data': tf.VarLenFeature(tf.float32),
            'doc2_bow_indices': tf.VarLenFeature(tf.int64),

            'doc_bert': tf.FixedLenFeature([768], tf.float32),
            'doc2_bert': tf.FixedLenFeature([768], tf.float32),

            'weak_hamming_dist': tf.FixedLenFeature([1], tf.int64),
            'doc_idx': tf.FixedLenFeature([1], tf.int64),
            'doc2_idx': tf.FixedLenFeature([1], tf.int64),
        }

        sample = tf.parse_single_example(data_record, features)

        for key in ['weak_hamming_dist', 'doc_idx', 'doc2_idx']:
            sample[key] = tf.squeeze(sample[key], -1)

        for key in ['weak_hamming_dist', 'doc_idx', 'doc2_idx']:
            sample[key] = tf.cast(sample[key], tf.int32)

        doc_bow_index = tf.expand_dims(sample['doc_bow_indices'].values, -1)# tf.reshape(sample['doc_bow_indices'].values, [-1, 1])
        doc_bow_values = sample['doc_bow_data'].values
        doc_bow = tf.sparse.SparseTensor(indices=doc_bow_index, values=doc_bow_values, dense_shape=[bowlen,])
        doc_bow = tf.sparse.to_dense(doc_bow, )

        doc2_bow_index =  tf.expand_dims(sample['doc2_bow_indices'].values, -1) # tf.reshape(sample['doc2_bow_indices'].values, [-1, 1])
        doc2_bow_values = sample['doc2_bow_data'].values
        doc2_bow = tf.sparse.SparseTensor(indices=doc2_bow_index, values=doc2_bow_values, dense_shape=[bowlen,])
        doc2_bow = tf.sparse.to_dense(doc2_bow)

        sample["doc_bow"] = doc_bow
        sample["doc2_bow"] = doc2_bow

        print(doc_bow, doc2_bow)
        feature_order = ['doc_bert', 'doc2_bert',
                         'weak_hamming_dist', 'doc_idx', 'doc2_idx',
                         'doc_bow', 'doc2_bow']

        return tuple([sample[key] for key in feature_order])

    output_t = [tf.int32 for _ in range(7)]
    output_t[0] = tf.float32
    output_t[1] = tf.float32
    output_t[-1] = tf.float32
    output_t[-2] = tf.float32
    output_t = tuple(output_t)

    bert_shape = tf.TensorShape([None,768,])
    default_shape = tf.TensorShape([None,])
    bow_shape = tf.TensorShape([None, bowlen,])
    output_s = [bert_shape, bert_shape, default_shape, default_shape, default_shape, bow_shape, bow_shape]
    output_s = tuple(output_s)

    dataset = tf.data.Dataset.from_tensor_slices(record_paths)
    if not is_test:
        dataset = dataset.repeat()
    dataset = dataset.shuffle(20)
    dataset = dataset.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.map(extract_fn, num_parallel_calls=3)
    if not is_test:
        dataset = dataset.shuffle(30000)
    dataset = dataset.batch(batchsize)
    dataset = dataset.prefetch(10)
    iterator = dataset.make_initializable_iterator()

    generic_iter = tf.data.Iterator.from_string_handle(handle, output_t, output_s)
    specific_handle = sess.run(iterator.string_handle())

    return specific_handle, iterator, generic_iter



def generator_nobert(sess, maxsize, handle, batchsize, record_paths, is_test, bowlen):
    def extract_fn(data_record):
        '''
            features["doc_input_ids"] = create_int_feature(doc_feature.input_ids)
            features["doc_input_mask"] = create_int_feature(doc_feature.input_mask)
            features["doc_segment_ids"] = create_int_feature(doc_feature.segment_ids)

            features["doc2_input_ids"] = create_int_feature(doc2_feature.input_ids)
            features["doc2_input_mask"] = create_int_feature(doc2_feature.input_mask)
            features["doc2_segment_ids"] = create_int_feature(doc2_feature.segment_ids)

            features["weak_hamming_dist"] = create_int_feature([int(weak_hamming_dist)])
            features["doc_idx"] = create_int_feature([int(doc_idx)])
            features["doc2_idx"] = create_int_feature([int(doc2_idx)])

            features["doc_bow"] = create_float_feature(doc_bow)
            features["doc2_bow"] = create_float_feature(doc2_bow)
        '''

        feature_order = ['doc_input_ids', 'doc_input_mask', 'doc_segment_ids',
                         'doc2_input_ids', 'doc2_input_mask', 'doc2_segment_ids',
                         'weak_hamming_dist', 'doc_idx', 'doc2_idx', 'doc_bow', 'doc_bow']

        features = {
            'doc_input_ids': tf.FixedLenFeature([maxsize], tf.int64),
            'doc_input_mask': tf.FixedLenFeature([maxsize], tf.int64),
            'doc_segment_ids': tf.FixedLenFeature([maxsize], tf.int64),

            'doc2_input_ids': tf.FixedLenFeature([maxsize], tf.int64),
            'doc2_input_mask': tf.FixedLenFeature([maxsize], tf.int64),
            'doc2_segment_ids': tf.FixedLenFeature([maxsize], tf.int64),

            'weak_hamming_dist': tf.FixedLenFeature([1], tf.int64),
            'doc_idx': tf.FixedLenFeature([1], tf.int64),
            'doc2_idx': tf.FixedLenFeature([1], tf.int64),

            'doc_bow': tf.FixedLenFeature([bowlen], tf.float32),
            'doc2_bow': tf.FixedLenFeature([bowlen], tf.float32),

        }

        sample = tf.parse_single_example(data_record, features)

        for key in ['weak_hamming_dist', 'doc_idx', 'doc2_idx']:
            sample[key] = tf.squeeze(sample[key], -1)

        for key in features.keys():
            if "bow" in key:
                continue
            sample[key] = tf.cast(sample[key], tf.int32)

        return tuple([sample[key] for key in feature_order])

    output_t = [tf.int32 for _ in range(11)]
    output_t[-2] = tf.float32
    output_t[-1] = tf.float32
    output_t = tuple(output_t)

    doc_shapes = tf.TensorShape([None,maxsize,])
    output_s = [doc_shapes for _ in output_t]
    output_s[-5] = tf.TensorShape([None,])
    output_s[-4] = tf.TensorShape([None,])
    output_s[-3] = tf.TensorShape([None,])
    output_s[-2] = tf.TensorShape([None,bowlen,])
    output_s[-1] = tf.TensorShape([None,bowlen,])

    output_s = tuple(output_s)

    dataset = tf.data.Dataset.from_tensor_slices(record_paths)
    if not is_test:
        dataset = dataset.repeat()
    dataset = dataset.shuffle(10)
    dataset = dataset.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.map(extract_fn, num_parallel_calls=3)
    if not is_test:
        dataset = dataset.shuffle(30000)
    dataset = dataset.batch(batchsize)
    dataset = dataset.prefetch(10)
    iterator = dataset.make_initializable_iterator()

    generic_iter = tf.data.Iterator.from_string_handle(handle, output_t, output_s)
    specific_handle = sess.run(iterator.string_handle())

    return specific_handle, iterator, generic_iter

