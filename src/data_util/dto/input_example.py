import logging
from dataclasses import dataclass
from typing import Optional, Union

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreTrainingInputExample(object):
  id: Optional[Union[str, int]]
  text_a: str
