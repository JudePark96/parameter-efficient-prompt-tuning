import logging
from typing import Optional, Dict

import torch
from torch import nn
from transformers import RobertaModel

from src.data_util.processors import Verbalizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class RobertaClassificationHead(nn.Module):
  """Head for sentence-level classification tasks."""

  def __init__(self, config, num_labels):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    classifier_dropout = (
      config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
    )
    self.dropout = nn.Dropout(classifier_dropout)
    self.out_proj = nn.Linear(config.hidden_size, num_labels)

  def forward(self, features, **kwargs):
    x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
    x = self.dropout(x)
    x = self.dense(x)
    x = torch.tanh(x)
    x = self.dropout(x)
    x = self.out_proj(x)
    return x


class ConventionalTuning(nn.Module):
  def __init__(self, model_name_or_config_path: str, task_name: str, freeze_lm: bool) -> None:
    super().__init__()
    self.roberta = RobertaModel.from_pretrained(model_name_or_config_path)
    self.cls_head = RobertaClassificationHead(self.roberta.config, len(Verbalizer[task_name]))

    if freeze_lm:
      for param in self.roberta.parameters():
        param.requires_grad = False

  def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    labels: Optional[torch.Tensor] = None
  ) -> Dict[str, torch.Tensor]:
    outputs = self.roberta(input_ids, attention_mask=attention_mask)
    logits = self.cls_head(outputs.last_hidden_state)

    loss = None
    if labels is not None:
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(logits, labels)

    return {
      'loss': loss,
      'logits': logits
    }
