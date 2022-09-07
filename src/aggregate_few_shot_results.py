import logging
import os.path
import sys

import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
  input_path = sys.argv[1]
  task = sys.argv[2]
  k_shot = sys.argv[3]
  seeds = sys.argv[4].split(',')

  if os.path.isdir(input_path) is False:
    logger.info(f'There is no test results yet: {input_path}')
    exit()

  dev_aggregated_results, test_aggregated_results = [], []
  dev_file_name, test_file_name = 'dev_best.txt', 'test_best.txt'

  for seed in seeds:
    path = os.path.join(input_path, f'{k_shot}-{seed}')

    # Dev Results
    with open(os.path.join(path, dev_file_name), 'r', encoding='utf-8') as f:
      for i in f:
        dev_aggregated_results.append(float(i))

    with open(os.path.join(path, test_file_name), 'r', encoding='utf-8') as f:
      for i in f:
        test_aggregated_results.append(float(i))

  dev_score, dev_std = np.mean(dev_aggregated_results), np.std(dev_aggregated_results)
  test_score, test_std = np.mean(test_aggregated_results), np.std(test_aggregated_results)

  logger.info(f'input path: {input_path}')
  logger.info(f'aggregated results will be written at: {input_path}')
  logger.info(f'task: {task}')
  logger.info(f'k_shot: {k_shot}')
  logger.info(f'dev mean score: {dev_score} | std: {dev_std}')
  logger.info(f'test mean score: {test_score} | std: {test_std}')

  with open(os.path.join(input_path, 'dev_aggregated.txt'), 'w', encoding='utf-8') as f:
    f.write(f'score: {str(dev_score)}' + '\n')
    f.write(f'std: {str(dev_std)}' + '\n')

  with open(os.path.join(input_path, 'test_aggregated.txt'), 'w', encoding='utf-8') as f:
    f.write(f'score: {str(test_score)}' + '\n')
    f.write(f'std: {str(test_std)}' + '\n')



