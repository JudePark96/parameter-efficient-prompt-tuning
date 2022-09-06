import argparse
import json
import logging
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaTokenizer, AdamW, get_linear_schedule_with_warmup

sys.path.append(os.getcwd() + "/../")  # noqa: E402

from src.modeling.modeling_prefix_tuning import PrefixRobertaModelForMaskedLM
from src.data_util.create_features import get_few_shot_prompt_dataloader
from src.data_util.processors import compute_metrics_mapping
from src.misc.checkpoint_utils import save_model_state_dict, write_log, write_score_log
from src.misc.common import print_args
from src.misc.gpu_utils.gpu_config import set_seed
from src.modeling.modeling_conventional import ConventionalTuning
from src.modeling.modeling_prompting import PromptingLM

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def sanity_checks(params):
  if os.path.isdir(params.save_checkpoints_dir):
    assert not os.listdir(params.save_checkpoints_dir), "checkpoint directory must be empty"
  else:
    os.makedirs(params.save_checkpoints_dir, exist_ok=True)


def _get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--device_ids", type=str, default="3",
                      help="comma separated list of devices ids in single node")
  parser.add_argument("--seed", type=int, default=203)
  parser.add_argument("--save_checkpoints_dir", type=str, default="checkpoints/")
  # parser.add_argument("--save_checkpoints_steps", type=int, default=10000)
  parser.add_argument("--log_step_count_steps", type=int, default=10)
  parser.add_argument("--num_train_epochs", type=int, default=5)
  parser.add_argument("--model_name_or_config_path", type=str, default='../resources/')
  parser.add_argument("--model_type", type=str, default='conventional')  # [conventional, prompt, prefix+prompt]
  parser.add_argument('--pre_seq_len', type=int, default=20, help='length of prepended prefix sequence.')
  parser.add_argument('--prefix_hidden_size', type=int, default=256,
                      help='number of hidden size of prefix encoder')
  parser.add_argument('--prefix_projection', default=False, action='store_true',
                      help='whether projecting the prefix by additional MLP layer.')
  parser.add_argument("--pretrained_prefix_path", type=str, default='../resources/')
  parser.add_argument('--initialize_from_vocab', default=False, action='store_true',
                      help='whether initializing embedding from pre-trained word embedding.')
  parser.add_argument('--freeze_lm', default=False, action='store_true',
                      help='whether freezing lm parameters when conventional tuning.')
  parser.add_argument("--input_path", type=str, default='')
  parser.add_argument("--task_name", type=str, default='')
  parser.add_argument('--train_batch_size', type=int, default=2)
  parser.add_argument("--eval_batch_size", type=int, default=8)
  parser.add_argument("--early_stopping", type=int, default=5)
  parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
  parser.add_argument("--learning_rate", type=float, default=5e-5)
  parser.add_argument("--warmup_proportion", type=float, default=0.06)
  parser.add_argument("--adam_beta1", type=float, default=0.9)
  parser.add_argument("--adam_beta2", type=float, default=0.999)
  parser.add_argument("--adam_epsilon", type=float, default=1e-6)
  parser.add_argument("--weight_decay", type=float, default=0.01)
  parser.add_argument("--max_grad_norm", type=float, default=1.0)
  parser.add_argument("--num_workers", type=int, default=30, help="number of workers")

  return parser


def count_parameters(model: nn.Parameter):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
  args = _get_parser().parse_args()
  args.n_gpu = 1
  set_seed(args)
  sanity_checks(args)

  args.task_name = args.task_name.lower()

  train_dataloader = get_few_shot_prompt_dataloader(args.input_path, data_type='train',
                                                    batch_size=args.train_batch_size,
                                                    num_workers=args.num_workers)

  dev_dataloader = get_few_shot_prompt_dataloader(args.input_path, data_type='dev', batch_size=args.eval_batch_size,
                                                  num_workers=args.num_workers)

  test_dataloader = get_few_shot_prompt_dataloader(args.input_path, data_type='test', batch_size=args.eval_batch_size,
                                                   num_workers=args.num_workers)

  if args.model_type == 'conventional':
    model = ConventionalTuning(args.model_name_or_config_path, args.task_name, args.freeze_lm)
  elif args.model_type == 'prompt':
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_config_path)
    model = PromptingLM(args.model_name_or_config_path, args.pre_seq_len, args.task_name, tokenizer,
                        args.initialize_from_vocab)
  elif args.model_type == 'prefix+prompt':
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_config_path)
    model = PrefixRobertaModelForMaskedLM.from_pretrained(args.model_name_or_config_path,
                                                          task=args.task_name,
                                                          tokenizer=tokenizer,
                                                          pre_seq_len=args.pre_seq_len,
                                                          prefix_hidden_size=args.prefix_hidden_size,
                                                          prefix_projection=args.prefix_projection,
                                                          training_type='finetuning')
    model.load_state_dict(torch.load(args.pretrained_prefix_path, map_location='cpu'))
  else:
    raise NotImplementedError(f'Not implemented type: {args.model_type}')

  logger.info(
    f'Total number of trainable parameters: {count_parameters(model)}')

  no_decay = ["bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [
    {
      "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
      "weight_decay": args.weight_decay
    },
    {
      "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
      "weight_decay": 0.0
    }
  ]

  optimizer = AdamW(optimizer_grouped_parameters,
                    lr=args.learning_rate,
                    betas=(args.adam_beta1, args.adam_beta2),
                    eps=args.adam_epsilon)

  t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
  warmup_steps = math.ceil(t_total * args.warmup_proportion)
  scheduler = get_linear_schedule_with_warmup(optimizer,
                                              num_warmup_steps=warmup_steps,
                                              num_training_steps=t_total)
  model.zero_grad()
  model.cuda()

  print_args(args)
  configuration = vars(args)
  save_configuration_path = os.path.join(args.save_checkpoints_dir, f"configuration.json")

  with open(save_configuration_path, "w") as fp:
    json.dump(configuration, fp, indent=2, ensure_ascii=False)

  logger.info("***** Running training *****")
  logger.info("  Num Epochs = %d", args.num_train_epochs)
  logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
  logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
  logger.info("  Total optimization steps = %d", t_total)
  global_steps = 0

  dev_best = 0.
  test_best = 0.
  early_stopping = 0

  for epoch in range(args.num_train_epochs):
    model.train()
    iter_loss, iter_veracity_loss, iter_evidence_loss = 0, 0, 0

    iter_bar = tqdm(train_dataloader, desc="Iter")
    for step, batch in enumerate(iter_bar):
      batch = tuple(i.cuda() for i in batch)
      output = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2])
      loss = output['loss']

      if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps
      loss.backward()

      if (step + 1) % args.gradient_accumulation_steps == 0:
        if args.max_grad_norm > 0:
          torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        global_steps += 1

        if global_steps % args.log_step_count_steps == 0:
          write_log(args.save_checkpoints_dir, "log_step.txt", iter_bar)

      iter_loss += loss.item()
      iter_bar.set_postfix({
        "epoch": f"{epoch}",
        "global_steps": f"{global_steps}",
        "learning_rate": f"{scheduler.get_last_lr()[0]:.10f}",
        "mean_loss": f"{iter_loss / (step + 1) * args.gradient_accumulation_steps:.5f}",
        "last_loss": f"{loss.item() * args.gradient_accumulation_steps:.5f}",
      })

    dev_score = eval(model, dev_dataloader, args, mode='Dev')['acc']
    logger.info(f'Dev Accuracy Score: {dev_score}')

    if dev_score > dev_best:
      early_stopping = 0
      dev_best = dev_score

      logger.info(f'Congratulations! New Dev Best Accuracy Score: {dev_best}')

      test_score = eval(model, test_dataloader, args, mode='Test')['acc']
      if test_score > test_best:
        test_best = test_score
        logger.info(f'Congratulations! New Test Best Accuracy Score: {test_best}')
        save_model_state_dict(args.save_checkpoints_dir,
                              f"{epoch}epoch_step{global_steps}_acc{test_best}.pth",
                              model)
    else:
      early_stopping += 1
      logger.info(f'Current Early Stopping States: {early_stopping}')

      if early_stopping >= args.early_stopping:
        logger.info(f'Early stopping starts!')
        break

  write_score_log(args.save_checkpoints_dir, "dev_best.txt", dev_best)
  write_score_log(args.save_checkpoints_dir, "test_best.txt", test_best)


def eval(model, loader: DataLoader, args, mode: str = 'Dev'):
  model.eval()

  preds = []
  trues = []

  for batch in tqdm(loader, desc=f'{mode} Iteration'):
    batch = tuple(t.cuda() for t in batch)

    with torch.no_grad():
      model_output = model(batch[0], attention_mask=batch[1])

      final_logits = F.softmax(model_output['logits'], dim=-1)
      final_logits = final_logits.argmax(dim=-1)

    final_logits = final_logits.detach().cpu().tolist()
    cls_logits = batch[-1].squeeze(dim=-1)
    cls_logits = cls_logits.detach().cpu().tolist()

    preds.extend(final_logits)
    trues.extend(cls_logits)

  output_dict = compute_metrics_mapping[args.task_name](args.task_name, np.array(preds), np.array(trues))
  model.train()

  return output_dict


if __name__ == '__main__':
  main()
