#### STS

Data Preprocessing 
```sh
$ python preprocess.py 
```
Corpus contextualization: Get word vectors for all words in vocabulary & WSD:
```sh
$ python contextualize.py --dataset_path dataset_path --temp_dir temp_dir --gpu_id 0
```
Get Sparse Document Vectors (SCDV) for documents and get similarity score
```sh
$ python SCDV.py
```
