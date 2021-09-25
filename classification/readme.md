#### 20Newsgroup
Change directory to 20news for experimenting on 20Newsgroup dataset and create train and test tsv files as follows:
```sh
$ cd 20news
$ python create_data.py
```
Get word vectors for all words in vocabulary through Word2Vec:
```sh
$ python contextualize.py --dataset_path dataset_path --temp_dir temp_dir --gpu_id 0
```
Get Sparse Document Vectors (SCDV) for documents in train and test set and accuracy of prediction on test set:
```sh
$ python SCDV.py
```

#### Other
Replace other with "amazon"/"twitter"/"bbcsport"/"recipel"/"classic"

Change directory:

```sh
$ cd other 
$ python preprocess.py
```
Get word vectors for all words in vocabulary through Word2Vec:
```sh
$ python contextualize.py --dataset_path dataset_path --temp_dir temp_dir --gpu_id 0
```
Get Sparse Document Vectors (SCDV) for documents in train and test set and accuracy of prediction on test set:
```sh
$ python SCDV.py
```
