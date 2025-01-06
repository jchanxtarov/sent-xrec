# Datasets to download

Download datasets (download xxx_exps.pkl.gz & put it into datasets/)
```
$ make load
```


# Original dataset structure

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