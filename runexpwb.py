'''
  Run experiment with wandb logging.

  Usage:
  python runexpwb.py --setting bag

  Note: wandb isn't compatible with running scripts in subdirs:
    e.g., python -m exps.chess.chessgfn
  So we call wandb init here.
'''
import torch
import wandb
import options
from attrdict import AttrDict

from exps.bag import bag
from exps.tfbind8 import tfbind8_oracle
from exps.tfbind10 import tfbind10
from exps.qm9str import qm9str
from exps.sehstr import sehstr

setting_calls = {
  'bag': lambda args: bag.main(args),
  'tfbind8': lambda args: tfbind8_oracle.main(args),
  'tfbind10': lambda args: tfbind10.main(args),
  'qm9str': lambda args: qm9str.main(args),
  'sehstr': lambda args: sehstr.main(args),
}


def main(args):
  print(f'Using {args.setting=} ...')
  exp_f = setting_calls[args.setting]
  exp_f(args)
  return


if __name__ == '__main__':
  args = options.parse_args()

  wandb.init(project=args.wandb_project,
             entity=args.wandb_entity,
             config=args, 
             mode=args.wandb_mode)
  args = AttrDict(wandb.config)
  args.run_name = wandb.run.name if wandb.run.name else 'None'

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'{device=}')
  args.device = device

  main(args)