import logging
from dataclasses import dataclass
from typing import List, Optional, Union

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputFeature(object):
  input_ids: List[int]
  attention_mask: List[int]
  token_type_ids: Optional[List[int]]
  label: Optional[Union[int, float]] = None


@dataclass(frozen=True)
class PreTrainingInputFeature(object):
  input_ids: List[int]
  attention_mask: List[int]
  token_type_ids: Optional[List[int]]
  label: Optional[Union[int, float]] = None


