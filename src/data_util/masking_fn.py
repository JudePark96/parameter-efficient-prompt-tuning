import collections
import logging
import random
from typing import List, Tuple

from transformers import RobertaTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def masked_language_modeling(
  tokens: List[str],
  vocab_words: List[str],
  rng: random,
  masked_lm_prob: float = .15,
  max_predictions_per_seq: int = 20,
  do_whole_word_mask: bool = False
) -> Tuple[List[str], List[int], List[str]]:
  cand_indexes = []
  for (i, token) in enumerate(tokens):
    # Modification for Roberta Language Model.
    if token == "<s>" or token == "</s>":
      continue
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word. When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.
    # if (do_whole_word_mask and len(cand_indexes) >= 1 and
    #   token.startswith("##")):
    if (do_whole_word_mask and len(cand_indexes) >= 1 and
      token.startswith("Ġ")):
      cand_indexes[-1].append(i)
    else:
      cand_indexes.append([i])

  rng.shuffle(cand_indexes)

  output_tokens = list(tokens)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index_set in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    if len(masked_lms) + len(index_set) > num_to_predict:
      continue
    is_any_index_covered = False
    for index in index_set:
      if index in covered_indexes:
        is_any_index_covered = True
        break
    if is_any_index_covered:
      continue
    for index in index_set:
      covered_indexes.add(index)

      masked_token = None
      # 80% of the time, replace with [MASK]
      if rng.random() < 0.8:
        masked_token = "<mask>"
      else:
        # 10% of the time, keep original
        if rng.random() < 0.5:
          masked_token = tokens[index]
        # 10% of the time, replace with random word
        else:
          masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

      output_tokens[index] = masked_token

      masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
  assert len(masked_lms) <= num_to_predict
  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return output_tokens, masked_lm_positions, masked_lm_labels


def salient_span_masking():
  pass


masking_funcs = {
  'ssm': salient_span_masking,
  'mlm': masked_language_modeling,
}


if __name__ == '__main__':
  masking_fn = masking_funcs['mlm']
  tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
  text = "Whole Word Masking means that if we mask all of the wordpieces corresponding to an original word. When a word has been split into WordPieces, the first token does not have any marker and any subsequence tokens are prefixed."
  print(masking_fn(tokenizer.tokenize(text), list(tokenizer.get_vocab().keys()), random.Random()))