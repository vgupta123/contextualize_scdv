##############################################
Major part of this code has been taken from https://github.com/dheeraj7596/ConWea
##############################################

import argparse
import json
import numpy as np
import flair, torch
from collections import defaultdict
from statistics import median
from sklearn.cluster import KMeans
from flair.data import Sentence
from flair.embeddings import BertEmbeddings
from flair.embeddings import TransformerWordEmbeddings
from nltk import sent_tokenize
from nltk.corpus import stopwords
import pandas as pd
from util import *

def main(dataset_path, temp_dir):
    def dump_bert_vecs(df, dump_dir):
        print("Getting BERT vectors...")
        embedding = TransformerWordEmbeddings('roberta-base',layers='-1')
        word_counter = defaultdict(int)
        stop_words = set(stopwords.words('english'))
        stop_words.add("would")
        except_counter = 0
        key = list(word_cnt.keys())

        for index, row in df.iterrows():
            file1 = open("progress.txt","w+")
            file1.write(str(index))
            print(index)
            if index % 100 == 0:
                print("Finished sentences: " + str(index) + " out of " + str(len(df)))
            line = row["news"]
            sentences = sent_tokenize(line)

            for sentence_ind, sent in enumerate(sentences):
                sentence = Sentence(sent, use_tokenizer=True)
                try:
                    embedding.embed(sentence)
                except Exception as e:
                    except_counter += 1
                    print("Exception Counter while getting BERT: ", except_counter, sentence_ind, index, e)
                    continue
                for token_ind, token in enumerate(sentence):
                    word = token.text
                    word = word.translate(str.maketrans('', '', string.punctuation))
                    if word in stop_words or "/" in word or len(word) == 0 or (word not in key) or word_cnt[word]<10:
                      #print("word")
                      continue
                    word_dump_dir = dump_dir + word
                    os.makedirs(word_dump_dir, exist_ok=True)
                    fname = word_dump_dir + "/" + str(word_counter[word]) + ".pkl"
                    word_counter[word] += 1
                    vec = token.embedding.cpu().numpy()
                    try:
                        with open(fname, "wb") as handler:
                            pickle.dump(vec, handler)
                    except Exception as e:
                        except_counter += 1
                        print("Exception Counter while dumping BERT: ", except_counter, sentence_ind, index, word, e)

    def compute_tau(label_seedwords_dict, bert_dump_dir):
        print("Computing Similarity Threshold..")
        seedword_medians = []
        for l in label_seedwords_dict:
            seed_words = label_seedwords_dict[l]
            for word in seed_words:
                try:
                    tok_vecs = read_bert_vectors(word, bert_dump_dir)
                    med = median(compute_pairwise_cosine_sim(tok_vecs))
                    seedword_medians.append(med)
                except Exception as e:
                    print("Exception: ", e)
        return median(seedword_medians)

    def cluster(tok_vecs, tau):
        def should_stop(cc):
            cos_sim = compute_pairwise_cosine_sim(cc)
            if (np.array(cos_sim) < tau).all():
                return False
            else:
                return True

        num_clusters = 2
        while True:
            if len(tok_vecs) < num_clusters:
                break
            km = KMeans(n_clusters=num_clusters, n_jobs=-1)
            km.fit(tok_vecs)
            cc = km.cluster_centers_
            if should_stop(cc):
                break
            num_clusters += 1

        num_clusters = num_clusters - 1
        if num_clusters == 1:
            cc = [np.mean(tok_vecs, axis=0)]
        elif len(tok_vecs) <= num_clusters:
            cc = tok_vecs
        else:
            km = KMeans(n_clusters=num_clusters, n_jobs=-1)
            km.fit(tok_vecs)
            cc = km.cluster_centers_
        return cc

    def cluster_words(tau, bert_dump_dir, cluster_dump_dir):
        print("Clustering words..")
        dir_set = get_relevant_dirs(bert_dump_dir)
        except_counter = 0
        print("Length of DIR_SET: ", len(dir_set))
        for word_index, word in enumerate(dir_set):
            if word_index % 100 == 0:
                print("Finished clustering words: " + str(word_index))
            try:
                tok_vecs = read_bert_vectors(word, bert_dump_dir)
                cc = cluster(tok_vecs, tau)
                word_cluster_dump_dir = cluster_dump_dir + word
                os.makedirs(word_cluster_dump_dir, exist_ok=True)
                with open(word_cluster_dump_dir + "/cc.pkl", "wb") as output_file:
                    pickle.dump(cc, output_file)
            except Exception as e:
                except_counter += 1
                print("Exception Counter while clustering: ", except_counter, word_index, e)

    def contextualize(df, cluster_dump_dir):
        def get_cluster(tok_vec, cc):
            max_sim = -10
            max_sim_id = -1
            for i, cluster_center in enumerate(cc):
                sim = cosine_similarity(tok_vec, cluster_center)
                if sim > max_sim:
                    max_sim = sim
                    max_sim_id = i
            return max_sim_id

        print("Contextualizing the corpus..")
        embedding = TransformerWordEmbeddings('roberta-base',layers='-1')
        stop_words = set(stopwords.words('english'))
        stop_words.add('would')
        except_counter = 0
        word_cluster = {}
        key = list(word_cnt.keys())

        for index, row in df.iterrows():
            file1 = open("progress.txt","w+")
            file1.write(str(index))
            if index % 100 == 0:
                print("Finished rows: " + str(index) + " out of " + str(len(df)))
            line = row["news"]
            sentences = sent_tokenize(line)
            for sentence_ind, sent in enumerate(sentences):
                sentence = Sentence(sent, use_tokenizer=True)
                embedding.embed(sentence)
                for token_ind, token in enumerate(sentence):
                    word = token.text
                    if word in stop_words:
                        continue
                    word_clean = word.translate(str.maketrans('', '', string.punctuation))
                    if len(word_clean) == 0 or word_clean in stop_words or "/" in word_clean or (word not in key) or word_cnt[word]<10:
                        continue
                    try:
                        cc = word_cluster[word_clean]
                    except:
                        try:
                            cc = word_cluster[word]
                        except:
                            word_clean_path = cluster_dump_dir + word_clean + "/cc.pkl"
                            word_path = cluster_dump_dir + word + "/cc.pkl"
                            try:
                                with open(word_clean_path, "rb") as handler:
                                    cc = pickle.load(handler)
                                word_cluster[word_clean] = cc
                            except:
                                try:
                                    with open(word_path, "rb") as handler:
                                        cc = pickle.load(handler)
                                    word_cluster[word] = cc
                                except Exception as e:
                                    except_counter += 1
                                    print("Exception Counter while getting clusters: ", except_counter, index, e)
                                    continue

                    if len(cc) > 1:
                        tok_vec = token.embedding.cpu().numpy()
                        cluster = get_cluster(tok_vec, cc)
                        sentence.tokens[token_ind].text = word + "$" + str(cluster)
                sentences[sentence_ind] = to_tokenized_string(sentence)
            df["news"][index] = " . ".join(sentences)
        return df, word_cluster

    pkl_dump_dir = dataset_path
    bert_dump_dir = temp_dir + "bert/"
    cluster_dump_dir = temp_dir + "clusters/"
    import pandas as pd
    df = pd.read_csv(pkl_dump_dir + "df.tsv", header=0, sep='\t')
    import pandas as pd
    import re
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np 
    def review_to_wordlist(review, remove_stopwords=False):
        #review = re.sub("[^a-zA-Z]", " ", review)
        words = review.split()
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops and len(w) > 1]
        return words
  
    traindata = []
    for i in range(0, len(df["news"])):
        traindata.append(review_to_wordlist(df["news"][i], True))

    word_cnt = {}
    for sentence in traindata:
        for word in sentence:
            word_cnt[word]= word_cnt.get(word,0)+1
    dump_bert_vecs(df, bert_dump_dir)
    tau = 0.80
    print("Cluster Similarity Threshold: ", tau)
    cluster_words(tau, bert_dump_dir, cluster_dump_dir)
    df_contextualized, word_cluster_map = contextualize(df, cluster_dump_dir)
    pickle.dump(df_contextualized, open(pkl_dump_dir + "df_contextualized.pkl", "wb"))
    pickle.dump(word_cluster_map, open(pkl_dump_dir + "word_cluster_map.pkl", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data/nyt/')
    parser.add_argument('--temp_dir', type=str, default='/tmp/')
    parser.add_argument('--gpu_id', type=str, default="cpu")
    args = parser.parse_args()
    if args.gpu_id != "cpu":
        flair.device = torch.device('cuda:' + str(args.gpu_id))
    main(dataset_path=args.dataset_path, temp_dir=args.temp_dir)
