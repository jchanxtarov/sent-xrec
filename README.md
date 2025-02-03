# Disentangling Likes and Dislikes in Personalized Generative Explainable Recommendation

- [Disentangling Likes and Dislikes in Personalized Generative Explainable Recommendation](#disentangling-likes-and-dislikes-in-personalized-generative-explainable-recommendation)
  - [Citation](#citation)
  - [Usage](#usage)
    - [models](#models)
    - [evaluations using task-specific metrics](#evaluations-using-task-specific-metrics)
  - [Datasets to download](#datasets-to-download)
  - [Original dataset structure](#original-dataset-structure)
  - [References](#references)
    - [Original datasets](#original-datasets)
    - [Benchmark model implementations](#benchmark-model-implementations)


## Citation
If you use our dataset in your work, please cite our paper:

> Ryotaro Shimizu, Takashi Wada, Yu Wang, Johannes Kruse, Sean O'Brien, Sai HtaungKham, Linxin Song, Yuya Yoshikawa, Yuki Saito, Fugee Tsung, Masayuki Goto, Julian McAuley. 2024. Disentangling Likes and Dislikes in Personalized Generative Explainable Recommendation. https://arxiv.org/abs/2410.13248

Bibtex:
```
@article{shimizu2024xrec,
  title={Disentangling Likes and Dislikes in Personalized Generative Explainable Recommendation},
  author={Ryotaro Shimizu and Takashi Wada and Yu Wang and Johannes Kruse and Sean O'Brien and Sai HtaungKham and Linxin Song and Yuya Yoshikawa and Yuki Saito and Fugee Tsung and Masayuki Goto and Julian McAuley},
  journal={arXiv preprint arXiv:2410.13248},
  year={2024}
}
```

## Usage

### models
0. Python Version
```
">=3.11,<3.13"
```
1. install poetry (if you need)
```
$ pip install poetry
```
2. library install & make virtual env
```
$ git clone git@github.com:jchanxtarov/sent_xrec_bench.git
$ cd sent_xrec_bench
$ make setup
```
3. download datasets (download xxx_exps.pkl.gz & put it into datasets/)
```
$ make load
```
4. run without logging
```
$ make dry-run {dataset_name} {model_name}
```
For example,
```
$ make dry-run ratebeer peter
```
1. 【optional】 create wandb account (if you need) and project 'sent_xrec_bench'
2. 【optional】 run with logging
```
$ make run {dataset_name} {model_name}
```
For example,
```
$ make run ratebeer peter
```

### evaluations using task-specific metrics

Please see the [notebook](https://github.com/jchanxtarov/sent_xrec_bench/blob/main/src/evals/evaluation.ipynb).

## Datasets to download
NB: Currently the download script is not supported yet.
*We are currently inquiring with Yelp regarding the possibility of releasing the dataset.

Download datasets (download xxx_exps.pkl.gz & put it into datasets/)
```
$ make load
```


## Original dataset structure

The original dataset is maintained in json format, and a row consists of the following:
```
{
  "item": "xxx",
  "user": "yyy",
  "rating": 5,
  "explanation": "hoge",
  "feature_pos": ["a", "b", "c"],
  "feature_neg": ["d", "e", "f"],
  "template": ("g", "h", "hoge"),
  "role": 0,
}
```

```
- item: item ID
- user: user ID
- rating: rating value
- explanation: the summarized text using LLM
- feature_pos: positive features list extracted from the explanation using LLM
- feature_neg: negative features list extracted from the explanation using LLM
- template: (randomly selected single word from feature_pos list, randomly selected single word from feature_neg list, explanation)
- role: the numbers 0, 1, and 2 represent training, validation, and testing, respectively.
```


## References

### Original datasets

- Amazon Movie [^1]
- Yelp [^2]
- RateBeer [^3]
  
[^1]: https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews
[^2]: https://www.yelp.com/dataset
[^3]: https://snap.stanford.edu/data/web-RateBeer.html


### Benchmark model implementations

- PETER, PETER+ [^4]: Lei Li, Yongfeng Zhang, and Li Chen. 2023. [Personalized Prompt Learning for Explainable Recommendation](https://dl.acm.org/doi/10.1145/3580488). ACM Transactions on Information Systems 41, 4, 1–26.
- ERRA [^5]: Hao Cheng, Shuo Wang, Wensheng Lu, Wei Zhang, Mingyang Zhou, Kezhong Lu, and Hao Liao. 2023. [Explainable Recommendation with Personalized Review Retrieval and Aspect Learning](https://aclanthology.org/2023.acl-long.4.pdf). In Proceedings of the Annual Meeting of the Association for Computational Linguistics. 51–64.
- CER [^6]: Jakub Raczyński, Mateusz Lango, and Jerzy Stefanowski. 2023. [The Problem of Coherence in Natural Language Explanations of Recommendations](https://arxiv.org/abs/2312.11356). In Proceedings of the European Conference on Artificial Intelligence.
- PEPLER, PEPLER-D [^7]: Lei Li, Yongfeng Zhang, and Li Chen. 2023. [Personalized Prompt Learning for Explainable Recommendation](https://dl.acm.org/doi/10.1145/3580488). ACM Transactions on Information Systems (TOIS).

[^4]: https://github.com/lileipisces/PETER
[^5]: https://github.com/Complex-data/ERRA
[^6]: https://github.com/JMRaczynski/CER
[^7]: https://github.com/lileipisces/PEPLER