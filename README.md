# Text Classification with Sparse Composite Document Vectors (SCDV)


## Introduction
  - For text classification and information retrieval tasks, text data has to be represented as a fixed dimension vector. 
  - We propose simple feature construction technique named [**SCDV: Sparse Composite Document Vectors using soft clustering over distributional representations.**](https://www.aclweb.org/anthology/D17-1069.pdf) presented at EMNLP 2017.
  - We demonstrate our method through experiments on multi-class classification on 20newsgroup dataset and multi-label text classification on Reuters-21578 dataset. 

## Citation
If you find SCDV useful in your research, please consider citing:
```
@misc{gupta2021unsupervised,
      title={Unsupervised Contextualized Document Representation}, 
      author={Ankur Gupta and Vivek Gupta},
      year={2021},
      eprint={2109.10509},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Testing
Please refer to folders classification, STS & Concept_Match which contains script and data related to experiments. 

## Requirements
Minimum requirements:
  -  Python 3.7
  -  NumPy 1.17.2
  -  Scikit-learn 0.23.1
  -  Pandas 0.25.1
  -  Gensim 3.8.1
  -  sgmllib3k
  -  flair 0.9

For theory and explanation of this work, please visit our [SustaiNLP paper](https://arxiv.org/pdf/2109.10509.pdf)
