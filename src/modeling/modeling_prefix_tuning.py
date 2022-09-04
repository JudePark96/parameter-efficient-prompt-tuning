import logging
from abc import ABC
from typing import Union

from transformers import RobertaPreTrainedModel, RobertaModel, RobertaTokenizer, AutoTokenizer
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.roberta.modeling_roberta import RobertaLMHead

from src.data_util.create_features import get_few_shot_prompt_dataloader, get_pretraining_dataloader
from src.data_util.processors import Verbalizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn


class PrefixEncoder(nn.Module):
  """
  The torch.nn model to encode the prefix
  Input shape: (batch-size, prefix-length)
  Output shape: (batch-size, prefix-length, 2*layers*hidden)
  """

  def __init__(self, config, pre_seq_len: int, prefix_hidden_size: int, prefix_projection: bool):
    super().__init__()
    self.prefix_projection = prefix_projection
    if self.prefix_projection:
      # Use a two-layer MLP to encode the prefix
      self.embedding = nn.Embedding(pre_seq_len, config.hidden_size)
      self.trans = nn.Sequential(
        nn.Linear(config.hidden_size, prefix_hidden_size),
        nn.Tanh(),
        nn.Linear(prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
      )
    else:
      self.embedding = torch.nn.Embedding(pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

  def forward(self, prefix: torch.Tensor):
    if self.prefix_projection:
      prefix_tokens = self.embedding(prefix)
      past_key_values = self.trans(prefix_tokens)
    else:
      past_key_values = self.embedding(prefix)
    return past_key_values


class PrefixRobertaModelForMaskedLM(RobertaPreTrainedModel, ABC):
  _keys_to_ignore_on_load_missing = [r"position_ids"]

  def __init__(self,
               config,
               pre_seq_len: int,
               prefix_hidden_size: int,
               prefix_projection: bool,
               task: str = None,
               tokenizer: Union[RobertaTokenizer, AutoTokenizer] = None,
               do_mlm: bool = True,
               training_type: str = 'pretraining'):
    super().__init__(config)
    self.roberta = RobertaModel(config, add_pooling_layer=False)

    for param in self.roberta.parameters():
      param.requires_grad = False

    self.pre_seq_len = pre_seq_len
    self.prefix_hidden_size = prefix_hidden_size
    self.prefix_projection = prefix_projection
    self.do_mlm = do_mlm
    self.training_type = training_type
    self.n_layer = config.num_hidden_layers
    self.n_head = config.num_attention_heads
    self.n_embd = config.hidden_size // config.num_attention_heads

    self.prefix_tokens = torch.arange(self.pre_seq_len).long()
    self.prefix_encoder = PrefixEncoder(config, self.pre_seq_len, self.prefix_hidden_size, self.prefix_projection)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

    if self.do_mlm:
      self.lm_head = RobertaLMHead(config)

    if self.training_type == 'finetuning':
      self.verbalizer = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word)) for word in Verbalizer[task]]
      self.mask_token_id = tokenizer.mask_token_id

    # compute the number of total parameters and tunable parameters
    total_param = sum(p.numel() for p in self.parameters())
    trainable_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print('total param is {}, trainable param is {}'.format(total_param, trainable_param))

  def get_prompt(self, batch_size: int):
    prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
    past_key_values = self.prefix_encoder(prefix_tokens)
    # bsz, seqlen, _ = past_key_values.shape
    past_key_values = past_key_values.view(
      batch_size,
      self.pre_seq_len,
      self.n_layer * 2,
      self.n_head,
      self.n_embd
    )
    past_key_values = self.dropout(past_key_values)
    past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
    return past_key_values

  def forward(
    self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
  ) -> MaskedLMOutput:
    return_dict = return_dict if return_dict is not None else self.roberta.config.use_return_dict

    past_key_values = self.get_prompt(batch_size=input_ids.shape[0])
    prefix_attention_mask = torch.ones(input_ids.shape[0], self.pre_seq_len).to(input_ids.device)
    attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

    outputs = self.roberta(
      input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      position_ids=position_ids,
      head_mask=head_mask,
      inputs_embeds=inputs_embeds,
      output_attentions=output_attentions,
      output_hidden_states=True,
      return_dict=True,
      past_key_values=past_key_values,  # new added
    )

    last_hidden_state = outputs.last_hidden_state
    prediction_scores = self.lm_head(last_hidden_state)

    if self.training_type == 'pretraining':
      masked_lm_loss = None
      if labels is not None:
        loss_fct = nn.CrossEntropyLoss()
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

      return MaskedLMOutput(
        loss=masked_lm_loss,
        logits=prediction_scores,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions
      )
    elif self.training_type == 'finetuning':
      loss_fct = nn.CrossEntropyLoss()
      masked_positions = (input_ids == self.mask_token_id).nonzero()
      masked_logits = torch.cat([prediction_scores[idx, position, :].unsqueeze(dim=0)
                                 for idx, position in enumerate(masked_positions[:, 1])], dim=0)
      verbalizer = torch.tensor(self.verbalizer).long().to(masked_logits.device).squeeze(dim=-1)
      verbalized_logits = masked_logits[:, verbalizer]
      loss = None
      if labels is not None:
        loss = loss_fct(verbalized_logits, labels)
      return {
        'logits': verbalized_logits,
        'loss': loss
      }
    else:
      raise ValueError(f'Invalid training type: {self.training_type}')


if __name__ == '__main__':
  loader = get_pretraining_dataloader('../../rsc/preprocessed/pretraining/',
                                      'test_roberta_maxlen384_prob0.15_max_pred_per_seq20_do_whole_word_maskTrue',
                                      4,
                                      16)

  model = PrefixRobertaModelForMaskedLM.from_pretrained('roberta-base',
                                                        pre_seq_len=20,
                                                        prefix_hidden_size=256,
                                                        prefix_projection=False)
  for batch in loader:
    print(model(batch[0], batch[1], labels=batch[2]))
    exit()
