import argparse
import logging
import multiprocessing
import os
import pickle
import random
import sys
from functools import partial
from random import Random
from typing import Union, Optional, List, Callable

from datasets import load_dataset
from tqdm import tqdm
from transformers import InputExample, AutoTokenizer, RobertaTokenizer

sys.path.append(os.getcwd() + "/../")  # noqa: E402

from src.data_util.dto.input_example import PreTrainingInputExample
from src.data_util.dto.input_feature import InputFeature, PreTrainingInputFeature
from src.data_util.masking_fn import masking_funcs
from src.data_util.processors import Sst2Processor, Templatizer, processors_mapping

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_options():
  options = argparse.ArgumentParser()
  options.add_argument('--input_path', type=str, default='')
  options.add_argument('--output_path', type=str, default='')
  options.add_argument('--task', type=str, default='sst-2')  # [finetune, pretraining]
  options.add_argument('--preprocessing_type', type=str, default='finetune')  # [finetune, pretraining]
  options.add_argument('--model_name_or_config_path', type=str, default='roberta-base')
  options.add_argument('--max_length', type=int, default=256)
  options.add_argument('--masked_lm_prob', type=float, default=.15)
  options.add_argument('--max_predictions_per_seq', type=int, default=20)
  options.add_argument('--do_whole_word_mask', action='store_true', default=False)
  options.add_argument('--seed', type=int, default=13)

  return options.parse_args()


def convert_examples_to_prompt_few_shot_features(
  task: str,
  model_type: str,
  max_length: int,
  tokenizer: Union[RobertaTokenizer, AutoTokenizer],
  example: InputExample
) -> InputFeature:
  """
  @param task:
  @param model_type:
  @param max_length:
  @param tokenizer:
  @param example:
  @return:
  """
  template = tokenizer.tokenize(Templatizer[task].format(tokenizer.mask_token))

  text_a = tokenizer.tokenize(example.text_a)
  if example.text_b is None:
    tokens = ([tokenizer.cls_token] + text_a + template)[:(max_length - 1)] + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)

    if model_type == 'bert':
      token_type_ids = [0] * len(input_ids)
      while len(token_type_ids) < max_length:
        token_type_ids.append(0)

    while len(input_ids) < max_length:
      input_ids.append(tokenizer.pad_token_id)
      attention_mask.append(0)

    return InputFeature(
      input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      label=example.label
    ) if model_type == 'bert' else InputFeature(
      input_ids=input_ids,
      attention_mask=attention_mask,
      label=example.label
    )
  else:
    text_b = tokenizer.tokenize(example.text_b)
    tokens = ([tokenizer.cls_token] + text_a + template + text_b)[:(max_length - 1)] + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)

    if model_type == 'bert':
      token_type_ids = [0] * len(input_ids)
      while len(token_type_ids) < max_length:
        token_type_ids.append(0)

    while len(input_ids) < max_length:
      input_ids.append(tokenizer.pad_token_id)
      attention_mask.append(0)

    return InputFeature(
      input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      label=example.label
    ) if model_type == 'bert' else InputFeature(
      input_ids=input_ids,
      attention_mask=attention_mask,
      label=example.label
    )


def convert_examples_to_mlm_pretraining_features(
  example: str,
  max_length: str,
  rng: Random,
  tokenizer: Union[RobertaTokenizer, AutoTokenizer],
  vocab_words: List[str],
  masking_func: Callable,
  model_type: str = 'roberta',
  masked_lm_prob: float = .15,
  max_predictions_per_seq: int = 20,
  do_whole_word_mask: bool = False
) -> PreTrainingInputFeature:
  """
  @param example:
  @param max_length:
  @param rng:
  @param tokenizer:
  @param vocab_words:
  @param masking_func:
  @param model_type:
  @param masked_lm_prob:
  @param max_predictions_per_seq:
  @param do_whole_word_mask:
  @return:
  """
  tokens = (['<s>'] + tokenizer.tokenize(example))[:(max_length - 1)] + ['</s>']

  # (['[CLS]', 'whole', 'word', 'mask', '##ing', 'means', 'that', 'if', ..., '[SEP]'],
  #  [9, 18, 21],
  #  ['mask', 'an', '.'])
  input_tokens, masked_lm_positions, masked_lm_labels = masking_func(tokens, vocab_words, rng,
                                                                     masked_lm_prob=masked_lm_prob,
                                                                     max_predictions_per_seq=max_predictions_per_seq,
                                                                     do_whole_word_mask=do_whole_word_mask)
  input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

  if model_type == 'bert':
    token_type_ids = [1] * len(input_ids)

  label_tokens = tokenizer.convert_tokens_to_ids(masked_lm_labels)
  labels = [-100] * len(input_ids)

  for iter_idx, position in enumerate(masked_lm_positions):
    labels[position] = label_tokens[iter_idx]

  attention_mask = [1] * len(input_ids)

  while len(input_ids) < max_length:
    # input_ids.append(tokenizer.pad_token_id)
    input_ids.append(1)
    attention_mask.append(0)
    labels.append(-100)

    if model_type == 'bert':
      token_type_ids.append(0)

  return PreTrainingInputFeature(
    input_ids=input_ids,
    attention_mask=attention_mask,
    token_type_ids=token_type_ids,
    label=labels
  ) if model_type == 'bert' else \
    PreTrainingInputFeature(
      input_ids=input_ids,
      attention_mask=attention_mask,
      label=labels
    )


if __name__ == '__main__':
  opt = get_options()
  logger.info(vars(opt))

  if opt.preprocessing_type == 'pretraining':
    with open(opt.input_path, 'r', encoding='utf-8') as f:
      raw_pretraining_data = [line.strip() for line in tqdm(f, desc='reading pretraining data ...')]

    tokenizer = AutoTokenizer.from_pretrained(opt.model_name_or_config_path)
    vocab_words = list(tokenizer.get_vocab().keys())
    masking_func = masking_funcs['mlm']
    n_threads = multiprocessing.cpu_count()

    raw_pretraining_data = raw_pretraining_data[:10000000]

    with multiprocessing.Pool(n_threads) as p:
      preprocessing_func = partial(convert_examples_to_mlm_pretraining_features, max_length=opt.max_length,
                                   masking_func=masking_func, vocab_words=vocab_words,
                                   rng=random.Random(opt.seed), tokenizer=tokenizer, model_type='roberta',
                                   masked_lm_prob=opt.masked_lm_prob,
                                   max_predictions_per_seq=opt.max_predictions_per_seq,
                                   do_whole_word_mask=opt.do_whole_word_mask)
      pretraining_data = list(tqdm(p.map(preprocessing_func, raw_pretraining_data, chunksize=16),
                                   total=len(raw_pretraining_data), desc='preprocessing pretraining data ...'))

    # pretraining_data = [preprocessing_func(raw) for raw in
    #                     tqdm(raw_pretraining_data, desc='preprocessing pretraining data ...')]

    with open(os.path.join(opt.output_path, f'roberta_maxlen${opt.max_length}_prob{opt.masked_lm_prob}'
                                            f'_max_pred_per_seq${opt.max_predictions_per_seq}_'
                                            f'do_whole_word_mask${opt.do_whole_word_mask}.pkl'), 'wb') as f:
      pickle.dump(pretraining_data, f)

  elif opt.preprocessing_type == 'finetune':
    os.makedirs(os.path.join(opt.output_path, opt.task.lower()), exist_ok=True)

    task_data = [processors_mapping[opt.task.lower()].get_train_examples(opt.input_path),
                 processors_mapping[opt.task.lower()].get_dev_examples(opt.input_path),
                 processors_mapping[opt.task.lower()].get_test_examples(opt.input_path)]

    tokenizer = AutoTokenizer.from_pretrained(opt.model_name_or_config_path)

    for t, name in zip(task_data, ['train', 'dev', 'test']):
      if name == 'test':
        preprocessed_task_data = [convert_examples_to_prompt_few_shot_features(opt.task.lower(), model_type='roberta',
                                                                               max_length=opt.max_length,
                                                                               tokenizer=tokenizer,
                                                                               example=i) for i in tqdm(t)]
      else:
        preprocessed_task_data = [convert_examples_to_prompt_few_shot_features(opt.task.lower(), model_type='roberta',
                                                                               max_length=opt.max_length,
                                                                               tokenizer=tokenizer,
                                                                               example=i) for i in t]
      with open(os.path.join(opt.output_path, f'{name}.pkl'), 'wb') as f:
        pickle.dump(preprocessed_task_data, f)
  else:
    raise ValueError(f'invalid {opt.preprocessing_type}')

  # examples = Sst2Processor().get_train_examples('../../rsc/k-shot/SST-2/16-100')
