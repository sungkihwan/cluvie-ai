import random
import logging

import torch
import numpy as np

from scipy.stats import pearsonr, spearmanr
from seqeval import metrics as seqeval_metrics
from sklearn import metrics as sklearn_metrics

from transformers import (
    ElectraConfig,
    ElectraTokenizer,
    ElectraForSequenceClassification,
    ElectraForTokenClassification,
    ElectraForQuestionAnswering,
)

CONFIG_CLASSES = {
    "koelectra-base-v3": ElectraConfig,
}

TOKENIZER_CLASSES = {
    "koelectra-base-v3": ElectraTokenizer,
}

MODEL_FOR_SEQUENCE_CLASSIFICATION = {
    "koelectra-base-v3": ElectraForSequenceClassification,
}

MODEL_FOR_TOKEN_CLASSIFICATION = {
    "koelectra-base-v3": ElectraForTokenClassification,
}

MODEL_FOR_QUESTION_ANSWERING = {
    "koelectra-base-v3": ElectraForQuestionAnswering,
}


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def simple_accuracy(labels, preds):
    return (labels == preds).mean()


def acc_score(labels, preds):
    return {
        "acc": simple_accuracy(labels, preds),
    }


def pearson_and_spearman(labels, preds):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def f1_pre_rec(labels, preds, is_ner=True):
    if is_ner:
        return {
            "precision": seqeval_metrics.precision_score(labels, preds, suffix=True),
            "recall": seqeval_metrics.recall_score(labels, preds, suffix=True),
            "f1": seqeval_metrics.f1_score(labels, preds, suffix=True),
        }
    else:
        return {
            "precision": sklearn_metrics.precision_score(labels, preds, average="macro"),
            "recall": sklearn_metrics.recall_score(labels, preds, average="macro"),
            "f1": sklearn_metrics.f1_score(labels, preds, average="macro"),
        }


def show_ner_report(labels, preds):
    return seqeval_metrics.classification_report(labels, preds, suffix=True)


def compute_metrics(task_name, labels, preds):
    assert len(preds) == len(labels)
    if task_name == "hate-speech":
        return f1_pre_rec(labels, preds, is_ner=False)
    elif task_name == "naver-ner":
        return f1_pre_rec(labels, preds, is_ner=True)
    else:
        raise KeyError(task_name)
