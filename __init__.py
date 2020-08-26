# __init__.py
import os
import requests
import zipfile
from pathlib import Path
import tensorflow as tf

from .bert_ner_define.index import define
from .bert_ner_define.bert import Ner

BASE_URL = 'http://ai-sample.oss-cn-hangzhou.aliyuncs.com/pipcook/models/'
MODEL_SMALL = os.path.join(BASE_URL, 'tf20-bert-base-cased/base.zip')
MODEL_LARGE = os.path.join(BASE_URL, 'tf20-bert-base-cased/large.zip')
BASE_PATH = os.path.join(str(Path.home()), '.pipcook', 'bert-model')

def download(url, filepath):
  r = requests.get(url, allow_redirects=True)
  with open(filepath, 'wb') as f:
    f.write(r.content)

def unZipData(src, dest):
  with zipfile.ZipFile(src, 'r') as zip_ref:
    zip_ref.extractall(dest)

def main(data, args):
  bertModel = 'base' if not hasattr(args, 'bertModel') else args.bertModel
  maxSeqLength = 128 if not hasattr(args, 'maxSeqLength') else args.maxSeqLength
  gpus = '0' if not hasattr(args, 'gpus') else args.gpus
  recoverPath = None if not hasattr(args, 'recoverPath') else args.recoverPath

  if not (bertModel == 'base' || bertModel == 'large'):
    raise Exception('bertModel must be base or large')

  modelPath = os.path.join(BASE_PATH, bertModel)
  modelDownloadUrl = MODEL_SMALL if bertModel == 'base' else MODEL_LARGE
  modelCheckpoint = os.path.join(modelPath, 'bert_model.ckpt.index')

  if os.path.exists(modelCheckpoint):
    download(modelDownloadUrl, os.path.join(BASE_PATH, bertModel + '.zip'))
    unZipData(os.path.join(BASE_PATH,  bertModel + '.zip'), BASE_PATH)
    os.remove(os.path.join(BASE_PATH,  bertModel + '.zip'))


  [ ner, strategy, loss_fct, max_seq_length, tokenizer ] = define({
    "bert_model": modelPath,
    "max_seq_length": maxSeqLength,
    "do_lower_case": False,
    "gpus": gpus
  })
  
  model = None
  if recoverPath is not None:
    model = Ner(recoverPath)
    ids = tf.ones([ 1, 128 ], dtype='tf.int64')
    ner(ids, ids, ids, ids, training=False)
    ner.load_weights(os.path.join(recoverPath, 'model.h5'))
  
  class PipcookModel:
    model = ner
    config = {
      "strategy": strategy,
      "loss_fct": loss_fct,
      "max_seq_length": max_seq_length,
      "tokenizer": tokenizer,
      "bert_model": recoverPath if recoverPath is not None else modelPath
    }
    def predict(self, inputData):
      output = self.model.predict(inputData.data)
      return output

  return PipcookModel()
  
  
  