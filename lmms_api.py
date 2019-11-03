"""
This file is meant to be a Python API for accessing WSD
"""
import atexit
import os
import subprocess
from types import SimpleNamespace
from zipfile import ZipFile

import gdown
import psutil
import spacy
import wget

from vectorspace import SensesVSM

ROOT = ".lmms"
# download source of bert checkpoint
BERT_URL = (
    "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip"
)
BERT_PATH = os.path.join(ROOT, "external", "bert")  # where to store checkpoint
# checkpoint name
BERT_CHECKPOINT = os.path.join(BERT_PATH, "cased_L-24_H-1024_A-16")

VECTORS_PATH = os.path.join(ROOT, "data", "vectors")
SV_DOWNLOAD_INFO = {
    "1024": SimpleNamespace(
        url="https://drive.google.com/uc?id=1kuwkTkSBz5Gv9CB_hfaBh1DQyC2ffKq0&export=download",
        filename="lmms_1024.bert-large-cased.npz",
    ),
    "2048": SimpleNamespace(
        url="https://drive.google.com/uc?id=15kJ8cY63wUwiMstHZ5wsX4_JFLnLJTjZ&export=download",
        filename="lmms_2048.bert-large-cased.npz",
    ),
    "2348": SimpleNamespace(
        url="https://drive.google.com/uc?id=1bwXfp-lUI91JBb5WE02ExAAHT-t3fhiN&export=download",
        filename="lmms_2348.bert-large-cased.fasttext-commoncrawl.npz",
    ),
}


class Disambiguator:
    def __init__(
        self,
        tokenizer=spacy.load("en_core_web_sm"),
        sv_size="1024",
        start_bert_server=True,
        bert_serving_start_path="bert-serving-start",
        bert_batch_size=32,
        keep_existing_server=False,
        kill_server_at_end=True
    ):
        """
        Creates the main interface to access LMMS word sense disambiguation.

        Starts the required bert_serving in the background if start_bert_server is True.

        Args:
            tokenizer: the tokenizer to use (default: spacy's tokenizer from "en_core_web_sm")
            sv_size (str): what size LMMS sense embedding vectors to use ("1024", "2048", or "2348") (default: "1024").
            start_bert_server (bool): whether to start bert_serving on init (default: True).
            bert_serving_start_path: the path to the `bert-serving-start` command. If using conda, it's probably in that envs' bin folder.
            bert_batch_size (int): batch size to start bert server with (default: 32)
            keep_existing_server (bool): whether to keep other BERT serving processes. Set to True if you prefer to manage your own external BERT serving (default: False).
            kill_server_at_end (bool): whether to auto kill the BERT serving process when Python exits. Can set to False if you'd like to keep the server alive between runs (default: True).
        """
        print("======== Setting Up LMMS =========")
        self.tokenizer = tokenizer

        sv_path = maybe_download_sense_vectors(sv_size)
        # the object that gets the senses
        self._senses_vsm = SensesVSM(sv_path, normalize=True)

        self._bert_process = None
        if start_bert_server:
            # stop others if they're already running, because they can conflict with
            # our own BERT serving
            if not keep_existing_server:
                kill_all_bert_serving()
            self._bert_serving_start_path = bert_serving_start_path
            self._set_bert_serving_start_command(bert_batch_size)
            self.maybe_start_bert_serving()
            if kill_server_at_end:
                atexit.register(self.kill_my_bert_serving)  # ensure cleanup
        print("======= Finished LMMS Setup =======")

    def sentence_to_synsets(self, sentence):
        """
        The primary function to convert a sentence to wordnet synset objects.
        Args:
            sentence (str): the sentence to parse
        Returns: a list of the top synsets for each token in the sentence
        """
        sent_info = self._parse_sent_info(sentence)

        if self._bert_process.poll() is not None:
            raise ValueError("BERT serving failed, cannot get disambiguation.")
        # get matches
        from exp_mapping import map_senses

        matches = map_senses(
            self._senses_vsm,
            sent_info["tokens"],
            sent_info["pos"],
            sent_info["lemmas"],
            use_lemma=False,
            use_postag=False,
        )

        def map_matches_to_synsets(match):
            """ Extracts the first sense match of a token, and converts the sensekey to
            a wordnet synset object. """
            first_match = match[0]
            sensekey, score = first_match
            from exp_mapping import wn_sensekey2synset

            synset = wn_sensekey2synset(sensekey)
            return synset, score

        synsets = list(map(map_matches_to_synsets, matches))
        return synsets

    def _parse_sent_info(self, sentence, merge_ents=False):
        """
        Parses the given sentence and tokenizes using spacy.
        Args:
            sentence (str): the english sentence to parse
            merge_ents: not sure what this does actually

        Returns: a dictionary of tokens, lemmas, pos, and the sentence
        """
        sent_info = {"tokens": [], "lemmas": [], "pos": [], "sentence": sentence}

        doc = self.tokenizer(sent_info["sentence"])

        if merge_ents:
            for ent in doc.ents:
                ent.merge()

        for tok in doc:
            sent_info["tokens"].append(tok.text.replace(" ", "_"))
            # sent_info['tokens'].append(tok.text)
            sent_info["lemmas"].append(tok.lemma_)
            sent_info["pos"].append(tok.pos_)

        sent_info["tokenized_sentence"] = " ".join(sent_info["tokens"])

        return sent_info

    def _set_bert_serving_start_command(self, batch_size):
        """ Sets the command to start BERT serving """
        self.start_bert_serving_command = (
            self._bert_serving_start_path,
            "-pooling_strategy",
            "NONE",
            "-model_dir",
            str(BERT_CHECKPOINT),
            "-pooling_layer",
            "-1",
            "-2",
            "-3",
            "-4",
            "-max_seq_len",
            "512",
            "-max_batch_size",
            str(batch_size),
            "-num_worker",
            "1",
            "-device_map",
            "0",
            "-cased_tokenization",
        )

    def maybe_start_bert_serving(self):
        """ Starts our own BERT serving process if BERT serving is not yet running """
        # download the checkpoint if need be
        maybe_download_pretrained_bert()
        # if bert-serving is not running:
        print("Checking if other BERT serving processes running...", end=" ")
        bs_pid = self.find_existing_bert_serving()
        if bs_pid is None:
            print("Found none.\nStarting up BERT serving...")
            self._bert_process = self._start_bert_wait_until_ready()
            print("BERT serving started and ready at PID", self._bert_process.pid)
        else:
            whose = "our own" if self._bert_process is not None else "external"
            print(f"Detected {whose} BERT serving already running, PID={bs_pid}")

    def _start_bert_wait_until_ready(self, verbose=False):
        """
        Creates a process that runs the bert serving command.
        Waits until the bert server reports that it's ready.
        Returns: the Process object created by subprocess
        """
        process = subprocess.Popen(
            self.start_bert_serving_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # wait for the process to be ready
        while True:
            # check that it hasn't died
            if process.poll() is not None:
                raise ValueError("BERT serving failed to start.")
            # if it hasn't died, we can read output
            line = process.stdout.readline().decode("utf-8")
            print(line, end="") if line and verbose else None
            if "all set, ready to serve" in line:
                return process

    def find_existing_bert_serving(self):
        """
        Return the first PID found if an existing BERT serving is running. Else return
        None.
        If we started up our own BERT serving, it will return our own first.
        """
        if self._bert_process is not None and self._bert_process.poll() is None:
            return self._bert_process.pid
        else:
            # otherwise check if bert-serving is running elsewhere out of our control
            for p in psutil.process_iter():
                if "bert-serving" in p.name():
                    try:
                        cmds = p.cmdline()
                        i = cmds.index("-model_dir")
                        if BERT_CHECKPOINT in cmds[i + 1]:
                            return p.pid
                    except (ValueError, IndexError):
                        continue
        return None

    def kill_my_bert_serving(self):
        """ Sends a kill() signal to our bert serving """
        print("Shutting down my BERT serving...")
        self._bert_process.kill()
        self._bert_process = None


def kill_all_bert_serving():
    """
    Search for and stop all BERT serving processes.
    Returns: True if processes found to stop, else False
    """
    found = False
    for p in psutil.process_iter():
        if "bert-serving" in p.name():
            found = True
            print("Killing", p.name(), ", PID=", p.pid)
            p.kill()
    if not found:
        print("No existing BERT servings found to kill.")
    return found


def maybe_download_sense_vectors(sensetype):
    """
    Downloads sense vectors if they haven't been downloaded already
    Args:
        sensetype: "1024", "2048" or "2348"
    Returns: the path of the downloaded .npz file
    """
    obj = SV_DOWNLOAD_INFO[sensetype]
    url, filename = obj.url, obj.filename
    out_path = os.path.join(VECTORS_PATH, filename)
    if not os.path.exists(out_path):
        os.makedirs(VECTORS_PATH, exist_ok=True)  # make dirs
        print(
            f"Downloading {sensetype}-LMMS vectors... This may take a while "
            f"(note: if you'd rather download it yourself, download {url} and extract "
            f"to {out_path})."
        )
        gdown.download(url, out_path, quiet=False)

    if os.path.exists(out_path):
        print(f"LMMS vectors {sensetype} found.")
    else:
        raise ValueError(f"failed to download {sensetype}-LMMS vectors to {out_path}")
    return out_path


def maybe_download_pretrained_bert():
    """
    Downloads BERT checkpoint if it's not already downloaded.
    >>> "mkdir .bert/"
    >>> "wget -P .bert/ https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip"
    >>> "unzip .bert/cased_L-24_H-1024_A-16.zip"
    """

    def is_downloaded():
        return os.path.exists(BERT_CHECKPOINT) and os.listdir(BERT_CHECKPOINT)

    if not is_downloaded():
        print(
            "Downloading BERT... This may take a while (note: if you'd rather download "
            f"it yourself, download {BERT_URL} and extract to {BERT_PATH})."
        )
        os.makedirs(BERT_PATH, exist_ok=True)
        # download the zip
        zip_file = BERT_CHECKPOINT + ".zip"
        wget.download(BERT_URL, out=zip_file)
        # extract the zip contents to the checkpoint
        with ZipFile(zip_file, "r") as z:
            z.extractall(BERT_PATH)
    # check again
    if is_downloaded():
        print("BERT checkpoint found.")
    else:
        raise ValueError("Failed to download BERT")
