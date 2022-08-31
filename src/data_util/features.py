import logging
from random import Random
from typing import Union, Optional

from datasets import load_dataset
from transformers import InputExample, AutoTokenizer, RobertaTokenizer

from src.data_util.dto.input_example import PreTrainingInputExample
from src.data_util.dto.input_feature import InputFeature, PreTrainingInputFeature
from src.data_util.masking_fn import masking_funcs
from src.data_util.processors import Sst2Processor, Templatizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_examples_to_few_shot_features(
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
  if example.text_b is not None:
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
  @param model_type:
  @param masked_lm_prob:
  @param max_predictions_per_seq:
  @param do_whole_word_mask:
  @return:
  """
  masking_func = masking_funcs['mlm']

  tokens = ([tokenizer.cls_token] + tokenizer.tokenize(example))[:(max_length - 1)] + [tokenizer.sep_token]
  vocab_words = list(tokenizer.get_vocab().keys())

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
    input_ids.append(tokenizer.pad_token_id)
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
      token_type_ids=token_type_ids,
      label=labels
    )


if __name__ == '__main__':
  a = load_dataset("wikipedia", "20220301.en")
  # print(a['train'][1])
  for i in a['train']:
    print(i['text'].split('\n\n'))
    exit()

  # examples = Sst2Processor().get_train_examples('../../rsc/k-shot/SST-2/16-100')
