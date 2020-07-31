

**Unsupervised Semantic Hashing with Pairwise Reconstruction**

**Citation:** Casper Hansen, Christian Hansen, Jakob Grue Simonsen, Stephen Alstrup, and Christina Lioma. 2020. Unsupervised Semantic Hashing with Pairwise Reconstruction. In _Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval_ (_SIGIR ’20_). Association for Computing Machinery, New York, NY, USA, 2009–2012.

Code is written for TensorFlow v. 1.12 and 1.14

You can download the datasets+code from: [https://www.dropbox.com/s/c2ekl15tb81fzjr/pairrec_data_code.zip?dl=0](https://www.dropbox.com/s/c2ekl15tb81fzjr/pairrec_data_code.zip?dl=0)
## Datasets
We evaluate our approach on 3 datasets: reuters, agnews, and TMC. In the data directory (data/orgdatasets/), we provide the train/val/test splits used in this paper, such that future work can re-use these for comparison of experimental results. Each dataset contains the following information in a pickle file: [train_indexes, val_indexes, test_indexes, BoW-vectors, labels]. You can use the \*\_indexes to index into the BoW-vectors and labels.

## Generating tfrecords prior to running the code
Make the following python calls from within the data directory:

    python make_tfrecords_sparse.py --dname reuters
    python make_tfrecords_sparse.py --dname TMC
    python make_tfrecords_sparse.py --dname agnews

## Running the code

main.py is the main file of the project. You can run it using the following parameters

- \-\-dname: the dataset name, e.g. "reuters"
- \-\-bits: the number of bits in the hash codes, e.g. 32
- \-\-numnbr: the number of pairs, e.g. 25 for reuters/TMC and 100 for agnews.

Additional parameters can be found in the main.py file and should be tuned for obtaining the best results. However, the default values will also work well.
