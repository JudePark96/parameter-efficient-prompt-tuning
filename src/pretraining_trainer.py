import argparse
import json
import logging
import math
import os
import shutil
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup, AdamW, Adafactor

sys.path.append(os.getcwd() + "/../")  # noqa: E402

from src.modeling.modeling_prefix_tuning import PrefixRobertaModelForMaskedLM
from src.misc.checkpoint_utils import write_log, save_model_state_dict
from src.data_util.create_features import get_pretraining_dataloader
from src.data_util.dto.input_feature import PreTrainingInputFeature
from src.misc.gpu_utils.gpu_config import set_seed, init_gpu_params
from src.misc.common import print_args

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
  parser.add_argument("--modeling_code_path", type=str, default='./modeling')
  parser.add_argument('--pre_seq_len', type=int, default=20, help='length of prepended prefix sequence.')
  parser.add_argument('--prefix_hidden_size', type=int, default=256,
                      help='number of hidden size of prefix encoder')
  parser.add_argument('--prefix_projection', default=False, action='store_true',
                      help='whether projecting the prefix by additional MLP layer.')
  parser.add_argument("--input_path", type=str, default='')
  parser.add_argument("--file_name", type=str, default='')
  parser.add_argument('--per_gpu_train_batch_size', type=int, default=2)
  parser.add_argument("--eval_batch_size", type=int, default=8)
  parser.add_argument("--dev_eval_step", type=int, default=2500)
  parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
  parser.add_argument("--learning_rate", type=float, default=5e-5)
  parser.add_argument("--optimizer", type=str, default='adafactor')
  parser.add_argument("--warmup_proportion", type=float, default=0.06)
  parser.add_argument("--adam_beta1", type=float, default=0.9)
  parser.add_argument("--adam_beta2", type=float, default=0.999)
  parser.add_argument("--adam_epsilon", type=float, default=1e-6)
  parser.add_argument("--weight_decay", type=float, default=0.01)
  parser.add_argument("--max_grad_norm", type=float, default=1.0)
  parser.add_argument("--num_workers", type=int, default=30, help="number of workers")
  parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
  parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs in the node.")

  return parser


def main():
  args = _get_parser().parse_args()

  set_seed(args)
  sanity_checks(args)
  init_gpu_params(args)

  if args.is_master:
    try:
      shutil.copytree(args.modeling_code_path, os.path.join(args.save_checkpoints_dir, 'modeling'))
      logger.info('Saving folder for reproduce the experiments.')
      logger.info(f"Saving folder: {os.path.join(args.save_checkpoints_dir, 'modeling')}")
    except Exception as e:
      logger.info(e)
      logger.info('Already exists modeling source codes.')

  train_dataloader, train_dataset = get_pretraining_dataloader(args.input_path, args.file_name,
                                                               batch_size=args.per_gpu_train_batch_size,
                                                               num_workers=args.num_workers, multi_gpu=args.multi_gpu)

  model = PrefixRobertaModelForMaskedLM.from_pretrained(args.model_name_or_config_path,
                                                        pre_seq_len=args.pre_seq_len,
                                                        prefix_hidden_size=args.prefix_hidden_size,
                                                        prefix_projection=args.prefix_projection)
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

  if args.optimizer == 'adafactor':
    optimizer = Adafactor(
      optimizer_grouped_parameters,
      lr=args.learning_rate,
      scale_parameter=False,
      relative_step=False,
    )
  elif args.optimizer == 'adamw':
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      betas=(args.adam_beta1, args.adam_beta2),
                      eps=args.adam_epsilon)
  else:
    raise ValueError(f'Invalid optimizer: {args.optimizer}')

  t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
  warmup_steps = math.ceil(t_total * args.warmup_proportion)
  scheduler = get_linear_schedule_with_warmup(optimizer,
                                              num_warmup_steps=warmup_steps,
                                              num_training_steps=t_total)

  model.zero_grad()
  model.cuda()

  if args.multi_gpu:
    model = DistributedDataParallel(model,
                                    device_ids=[args.device_ids[args.local_rank]],
                                    output_device=args.device_ids[args.local_rank],
                                    find_unused_parameters=True)

  if args.is_master:
    print_args(args)
    configuration = vars(args)
    save_configuration_path = os.path.join(args.save_checkpoints_dir, f"configuration.json")

    with open(save_configuration_path, "w") as fp:
      json.dump(configuration, fp, indent=2, ensure_ascii=False)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
      "  Total train batch size (w. parallel, distributed & accumulation) = %d",
      args.per_gpu_train_batch_size
      * args.gradient_accumulation_steps
      * (torch.distributed.get_world_size() if args.multi_gpu else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

  global_steps = 0

  for epoch in range(args.num_train_epochs):
    model.train()
    iter_loss, iter_veracity_loss, iter_evidence_loss = 0, 0, 0

    iter_bar = tqdm(train_dataloader, desc="Iter", disable=not args.is_master)
    for step, batch in enumerate(iter_bar):
      batch = tuple(i.cuda() for i in batch)
      output = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2])
      loss = output.loss

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

        if global_steps % args.log_step_count_steps == 0 and args.is_master:
          write_log(args.save_checkpoints_dir, "log_step.txt", iter_bar)

        if global_steps % args.dev_eval_step == 0 and args.is_master:
          save_model_state_dict(args.save_checkpoints_dir,
                                f"{epoch}epoch_step{(step + 1) * args.gradient_accumulation_steps}.pth", model)

      iter_loss += loss.item()
      iter_bar.set_postfix({
        "epoch": f"{epoch}",
        "global_steps": f"{global_steps}",
        "learning_rate": f"{scheduler.get_last_lr()[0]:.10f}",
        "mean_loss": f"{iter_loss / (step + 1) * args.gradient_accumulation_steps:.5f}",
        "last_loss": f"{loss.item() * args.gradient_accumulation_steps:.5f}",
      })

    if args.is_master:
      save_model_state_dict(args.save_checkpoints_dir, f"{epoch}epoch.pth", model)


if __name__ == "__main__":
  main()
