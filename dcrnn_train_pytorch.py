from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
import numpy as np
import os.path as osp

from lib.utils import load_graph_data
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor


def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)
        for year in range(int(supervisor_config['data']['begin_year']), int(supervisor_config['data']['end_year'])+1):
                adj_mx = np.load(osp.join(supervisor_config['data']['graph_pkl_filename'], str(year)+"_adj.npz"))["x"]
                supervisor = DCRNNSupervisor(adj_mx=adj_mx, year=year, **supervisor_config)
                supervisor.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)
