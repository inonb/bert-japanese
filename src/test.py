# coding=utf-8
# This file is based on https://github.com/google-research/bert/blob/master/run_classifier.py.
# It is changed to use SentencePiece tokenizer and https://www.rondhuit.com/download/ldcc-20140209.tar.gz.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import configparser
import csv
import json
import os
import sys
import tempfile
from . import tokenization_sentencepiece as tokenization
import tensorflow as tf
from . import utils

