# LMMS Python API for Automatic Word Sense Disambiguation

This repository is a fork of [Language Modeling Makes Sense (LMMS)](https://github.com/danlou/LMMS) for 
word-sense disambiguation (WSD). This repository provides a simple Python API wrapper to 
make using LMMS' WSD easier. It handles downloading the sense-vector data and BERT checkpoint, 
as well as managing bert-as-service in the background.

Usage is as simple as:
```python
from lmms_api import Disambiguator

dis = Disambiguator()

my_sentence = "My dog Spot looks pretty cool jumping over the bank."

# returns a list of top WordNet synsets and their probabilities
dis.sentence_to_synsets(my_sentence)

# outputs: [ (Synset('furthermore.r.01'), 0.73498523), (Synset('dog.n.01'), 0.88388115), 
#    (Synset('person.n.01'), 0.6367209), (Synset('look.v.02'), 0.80680996), 
#    (Synset('reasonably.r.01'), 0.83558476), (Synset('nice.a.01'), 0.6966169), 
#    (Synset('jump.v.01'), 0.79882085), (Synset('overboard.r.02'), 0.7287618), 
#    (Synset('hit_the_dirt.v.01'), 0.6825616), (Synset('bank.n.01'), 0.7973524) ]
```

On first run, constructing the Disambiguator object downloads the sense vectors 
and BERT checkpoint.

## Install
Run `pip install -r requirements.txt` (technically optional, but this makes sure all dependencies install correctly first).

Then run `pip install git+ssh://git@github.com/andrewjong/LMMS-API#egg=lmms_api` to allow importing 
`lmms_api`.

Note: the fastText dependency can get a little finicky. If installing through `requirements.txt` doesn't work for you, try [installing it separately yourself](https://github.com/facebookresearch/fastText#requirements).

## Further Options
The `Disambiguator()` constructor can take many arguments.

Change the argument `tokenizer` to change which tokenizer to use. Default is `spacy.load("en_core_web_sm")`.

Change the argument `sv_size` to choose a different sense-vector-embedding size to download. 
Available options are "1024", "2048", and "2348" (strings). Default is "1024".


### Managing bert-as-service
By default, constructing a Disambiguator object starts a new bert-as-service process in
 the background (`bert-serving-start`). Any existing `bert-serving-start` processes are 
 killed to prevent conflict. When the program ends, the created process is killed.
 However, this adds some overhead.
 
If you'd like to manage your own `bert-serving-start` process and not start one through this
API, pass `start_bert_server=False` to `Disambiguator()`.
Then, to run bert-as-service yourself, use:
```bash
$ bert-serving-start -pooling_strategy NONE -model_dir external/bert/cased_L-24_H-1024_A-16 \
  -pooling_layer -1 -2 -3 -4 -max_seq_len 512 -max_batch_size 32 -num_worker=1 \
  -device_map 0 -cased_tokenization
```

If you prefer not to kill other `bert-serving-start` processes, pass `keep_existing_server=True` to `Disambiguator()`.

If you'd like to keep the server alive after the program ends, pass `kill_server_at_end=False` to `Disambiguator()`.

Kill all bert-as-service processes with:
`import lmms_api; lmms_api.kill_all_bert_serving()`
# References

### ACL 2019

Main paper about LMMS ([arXiv](https://arxiv.org/abs/1906.10007)).

```
@inproceedings{loureiro-jorge-2019-language,
    title = "Language Modelling Makes Sense: Propagating Representations through {W}ord{N}et for Full-Coverage Word Sense Disambiguation",
    author = "Loureiro, Daniel  and
      Jorge, Al{\'\i}pio",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1569",
    doi = "10.18653/v1/P19-1569",
    pages = "5682--5691"
}
```

### SemDeep-5 at IJCAI 2019

Application of LMMS for the Word-in-Context (WiC) Challenge ([arXiv](https://arxiv.org/abs/1906.10002)).

```
@inproceedings{Loureiro2019LIAADAS,
  title={LIAAD at SemDeep-5 Challenge: Word-in-Context (WiC)},
  author={Daniel Loureiro and Al{\'i}pio M{\'a}rio Jorge},
  booktitle={SemDeep@IJCAI},
  year={2019}
}
```
