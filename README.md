# LMMS Python API for Automatic Word Sense Disambiguation

This repository is a fork of [Language Modeling Makes Sense (LMMS)](https://github.com/danlou/LMMS) for 
word-sense disambiguation (WSD). This repository provides a simple Python API wrapper to 
make using LMMS' WSD easier. It handles downloading the sense-vector data and BERT checkpoint, 
as well as managing bert-as-service in the background.

Usage is as simple as:
```python
from lmms_api import Disambiguator

dis = Disambiguator()

my_sentence = "My dog Spot looked pretty cool jumping over the bank."
dis.sentence_to_synsets(my_sentence)  # returns a list of top WordNet synsets
```
On first run, constructing the Disambiguator object will download the sense vectors 
and BERT checkpoint.

## Install
Run `pip install -r requirements.txt`.

Then run `pip install git+ssh://git@github.com/andrewjong/LMMS-API#egg=lmms_api` to allow importing 
`lmms_api`.

## Further Options
The Disambiguator() constructor can take many arguments.

Change the argument `tokenizer` to change which tokenizer to use. Default is `spacy.load("en_core_web_sm")`.

Change the argument `sv_size` to choose a different sense-vector-embedding size to download. 
Available options are "1024", "2048", and "2348" (strings). Default is "1024".


### Managing bert-as-service
By default, constructing a Disambiguator object starts a new bert-as-service process in
 the background (`bert-serving-start`). Any existing `bert-serving-start` processes are 
 killed to prevent conflict. When the program ends, the created process is killed.
 
If you'd like to manage your own bert-as-service process and not start one through this
API, pass `start_bert_server=False` to the constructor.

If you prefer not to kill other bert-as-service processes, pass `keep_existing_server=True`.

If you'd like to keep the server alive after the program ends, pass `kill_server_at_end=False`.

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
