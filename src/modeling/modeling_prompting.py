import logging
import os
from pathlib import Path
from typing import Optional, Union, Dict

import numpy as np
import torch
import torch.nn as nn
from transformers import BertForMaskedLM, RobertaForMaskedLM, RobertaTokenizer, AutoTokenizer

from src.data_util.create_features import get_few_shot_prompt_dataloader
from src.data_util.processors import Verbalizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptTuningMixIn:
  @classmethod
  def from_pretrained(
    cls,
    model_name_or_config_path: str,
    soft_prompt_path: str = None,
    n_tokens: int = None,
    initialize_from_vocab: bool = True,
    random_range: float = 0.5,
    **kwargs,
  ):
    model = super().from_pretrained(model_name_or_config_path, **kwargs)

    for params in model.parameters():
      params.required_grad = False

    if soft_prompt_path is not None:
      model.set_soft_prompt_embeds(soft_prompt_path)
    elif n_tokens is not None:
      logger.info("Initializing soft prompt...")
      model.initialize_soft_prompt(
        n_tokens=n_tokens,
        initialize_from_vocab=initialize_from_vocab,
        random_range=random_range,
      )

    return model

  def set_soft_prompt_embeds(
    self,
    soft_prompt_path: str,
  ) -> None:
    """
    Args:
        soft_prompt_path: torch soft prompt file path
    """
    self.soft_prompt = torch.load(
      soft_prompt_path, map_location=torch.device("cpu")
    )
    self.n_tokens = self.soft_prompt.num_embeddings
    print(f"Set soft prompt! (n_tokens: {self.n_tokens})")

  def initialize_soft_prompt(
    self,
    n_tokens: int = 20,
    initialize_from_vocab: bool = True,
    random_range: float = 0.5,
  ) -> None:
    self.n_tokens = n_tokens
    # following Lester et al. 2021 in initializing using the top 5000 random vocabs
    self.indices = np.random.permutation(range(5000))[:self.n_tokens]
    if initialize_from_vocab:
      init_prompt_value = nn.Parameter(self.get_input_embeddings().weight[self.indices].clone(), requires_grad=True)
    else:
      init_prompt_value = torch.FloatTensor(self.n_tokens, self.config.hidden_size).uniform_(
        -random_range, random_range
      )
    self.soft_prompt = nn.Embedding(n_tokens, self.config.hidden_size)
    # Initialize weight
    self.soft_prompt.weight = nn.parameter.Parameter(init_prompt_value, requires_grad=True)

  def _cat_learned_embedding_to_input(self, input_ids) -> torch.Tensor:
    inputs_embeds = self.get_input_embeddings()(input_ids)

    if len(list(inputs_embeds.shape)) == 2:
      inputs_embeds = inputs_embeds.unsqueeze(0)

    # [batch_size, n_tokens, n_embd]
    learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)

    inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

    return inputs_embeds

  def _extend_labels(self, labels, ignore_index=-100) -> torch.Tensor:
    if len(list(labels.shape)) == 1:
      labels = labels.unsqueeze(0)

    n_batches = labels.shape[0]
    return torch.cat(
      [
        torch.full((n_batches, self.n_tokens), ignore_index).to(self.device),
        labels,
      ],
      dim=1,
    )

  def _extend_attention_mask(self, attention_mask):

    if len(list(attention_mask.shape)) == 1:
      attention_mask = attention_mask.unsqueeze(0)

    n_batches = attention_mask.shape[0]
    return torch.cat(
      [torch.full((n_batches, self.n_tokens), 1).to(self.device), attention_mask],
      dim=1,
    )

  def save_soft_prompt(self, path: str, filename: str = "soft_prompt.model"):
    Path(path).mkdir(parents=True, exist_ok=True)
    torch.save(self.soft_prompt, os.path.join(path, filename))
    # print(f"Saved soft prompt: {os.path.join(path, filename)}")

  def forward(
    self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    inputs_embeds=None,
    return_dict=None,
  ):
    if input_ids is not None:
      inputs_embeds = self._cat_learned_embedding_to_input(input_ids).to(
        self.device
      )

    if attention_mask is not None:
      attention_mask = self._extend_attention_mask(attention_mask).to(self.device)

    # Drop most of the args for now
    return super().forward(
      attention_mask=attention_mask,
      inputs_embeds=inputs_embeds,
      return_dict=return_dict,
    )


class BertPromptingLM(PromptTuningMixIn, BertForMaskedLM):
  def __init__(self, config):
    super().__init__(config)


class RobertaPromptingLM(PromptTuningMixIn, RobertaForMaskedLM):
  def __init__(self, config):
    super().__init__(config)


class NormalPromptingLM(nn.Module):

  def __init__(self, model_name_or_config_path: str, n_tokens: int, task: str,
               tokenizer: Union[RobertaTokenizer, AutoTokenizer], initialize_from_vocab: bool = True) -> None:
    super().__init__()
    self.task = task
    self.model_name_or_config_path = model_name_or_config_path
    self.n_tokens = n_tokens
    self.initialize_from_vocab = initialize_from_vocab
    self.lm = RobertaForMaskedLM.from_pretrained(self.model_name_or_config_path)
    self.verbalizer = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word)) for word in Verbalizer[task]]
    self.mask_token_id = tokenizer.mask_token_id
    self.tokenizer = tokenizer
    self.loss_fn = nn.CrossEntropyLoss()

  def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    label: Optional[torch.Tensor]
  ) -> Dict[str, torch.Tensor]:
    masked_positions = (input_ids == self.mask_token_id).nonzero()
    lm_output = self.lm(input_ids, attention_mask)
    logits = lm_output.logits
    masked_logits = torch.cat([logits[idx, position, :].unsqueeze(dim=0)
                               for idx, position in enumerate(masked_positions[:, 1])], dim=0)
    verbalizer = torch.tensor(self.verbalizer).long().to(masked_logits.device).squeeze(dim=-1)
    verbalized_logits = masked_logits[:, verbalizer]
    loss = self.loss_fn(verbalized_logits, label)

    return {
      'logits': verbalized_logits,
      'loss': loss
    }


class PromptingLM(nn.Module):

  def __init__(self, model_name_or_config_path: str, n_tokens: int, task: str,
               tokenizer: Union[RobertaTokenizer, AutoTokenizer], initialize_from_vocab: bool = True) -> None:
    super().__init__()
    self.task = task
    self.model_name_or_config_path = model_name_or_config_path
    self.n_tokens = n_tokens
    self.initialize_from_vocab = initialize_from_vocab

    self.lm = RobertaPromptingLM.from_pretrained(self.model_name_or_config_path,
                                                 n_tokens=self.n_tokens,
                                                 initialize_from_vocab=self.initialize_from_vocab)
    self.verbalizer = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word)) for word in Verbalizer[task]]
    self.mask_token_id = tokenizer.mask_token_id
    self.tokenizer = tokenizer
    self.loss_fn = nn.CrossEntropyLoss()

  def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    label: Optional[torch.Tensor]
  ) -> Dict[str, torch.Tensor]:
    masked_positions = (input_ids == self.mask_token_id).nonzero()
    lm_output = self.lm(input_ids, attention_mask)
    logits = lm_output.logits

    masked_positions[:, 1] += self.n_tokens
    masked_logits = torch.cat([logits[idx, position, :].unsqueeze(dim=0)
                               for idx, position in enumerate(masked_positions[:, 1])], dim=0)
    verbalizer = torch.tensor(self.verbalizer).long().to(masked_logits.device).squeeze(dim=-1)
    verbalized_logits = masked_logits[:, verbalizer]

    loss = self.loss_fn(verbalized_logits, label)

    return {
      'logits': verbalized_logits,
      'loss': loss
    }


if __name__ == '__main__':
  loader = get_few_shot_prompt_dataloader('../../rsc/preprocessed/finetuning/SST-2/16-13', 'train', 4, 16)
  model = PromptingLM('roberta-base', 20, 'sst-2', RobertaTokenizer.from_pretrained('roberta-base'), True)
  for batch in loader:
    print(model(batch[0], batch[1], batch[2]))
    exit()
