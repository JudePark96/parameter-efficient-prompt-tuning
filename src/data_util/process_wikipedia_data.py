"""https://github.com/yxuansu/TaCL/blob/main/pretraining_data/english/funcs.py"""

import logging
import multiprocessing
from functools import partial
from typing import Dict, List

from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

import re

alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"


def split_into_sentences(text: str) -> str:
  text = " " + text + "  "
  text = text.replace("\n", " ")
  text = re.sub(prefixes, "\\1<prd>", text)
  text = re.sub(websites, "<prd>\\1", text)
  if "Ph.D" in text:
    text = text.replace("Ph.D.", "Ph<prd>D<prd>")
  text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
  text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
  text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
  text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
  text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
  text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
  text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)

  if "”" in text:
    text = text.replace(".”", "”.")
  if "\"" in text:
    text = text.replace(".\"", "\".")
  if "!" in text:
    text = text.replace("!\"", "\"!")
  if "?" in text:
    text = text.replace("?\"", "\"?")

  text = text.replace(".", ".<stop>")
  text = text.replace("?", "?<stop>")
  text = text.replace("!", "!<stop>")
  text = text.replace("<prd>", ".")
  sentences = text.split("<stop>")
  sentences = sentences[:-1]
  sentences = [s.strip() for s in sentences]
  return sentences


def fetch_lines(item: Dict[str, str], stop_prefix_list: List[str]) -> List[str]:
  article = item['text']
  article_list = article.split('\n')
  res_list = []
  break_flag = False
  for text in article_list:
    for prefix in stop_prefix_list:
      if text.startswith(prefix):
        break_flag = True

    if len(text.split()) < 3:
      pass
    else:
      res_list.append(text)
    if break_flag:
      break

  return res_list


def process_lines(item: str) -> List[str]:
  sentence_list = []
  for text in item:
    one_sen_list = split_into_sentences(text)
    for sen in one_sen_list:
      if len(sen) < 3:
        pass
      else:
        sentence_list.append(sen)
  return sentence_list


if __name__ == '__main__':
  from datasets import load_dataset

  dataset = load_dataset('wikipedia', "20220301.en", split='train')
  stop_prefix_list = ['References', 'External links', 'Category:', 'See also']

  n_threads = multiprocessing.cpu_count() // 2

  with multiprocessing.Pool(n_threads) as p:
    fetching_func = partial(fetch_lines, stop_prefix_list=stop_prefix_list)
    fetched_lines = list(tqdm(p.map(fetching_func, dataset, chunksize=64), total=len(dataset), desc='fetching lines'))
    process_func = partial(process_lines)
    processed_lines = list(tqdm(p.map(process_func, fetched_lines, chunksize=64), total=len(dataset),
                                desc='processing lines'))

    all_doc_list = []

    for doc in tqdm(processed_lines, desc='filtering doc by list length'):
      if len(doc) < 2:
        continue
      all_doc_list += [doc]

    out_f = './eng_wiki.txt'
    with open(out_f, 'w', encoding='utf8') as o:
      for doc in all_doc_list:
        for sen in doc:
          o.writelines(sen + '\n')
        o.writelines('\n')
