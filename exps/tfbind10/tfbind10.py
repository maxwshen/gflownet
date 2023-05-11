'''
  TFBind8
  Oracle
  Start from scratch
  No proxy
'''
import copy, pickle
import numpy as np
from polyleven import levenshtein

import gflownet.trainers as trainers
from gflownet.GFNs import models
from gflownet.MDPs import seqpamdp, seqinsertmdp, seqarmdp
from gflownet.monitor import TargetRewardDistribution, Monitor

def dynamic_inherit_mdp(base, args):

  class TFBind10MDP(base):
    def __init__(self, args):
      super().__init__(args,
                       alphabet=list('0123'),
                       forced_stop_len=10)
      self.args = args

      # Read from file
      print(f'Loading data ...')
      with open('datasets/tfbind10/tfbind10-exact-v0-all.pkl', 'rb') as f:
        oracle_d = pickle.load(f)
      
      munge = lambda x: ''.join([str(c) for c in list(x)])
      self.oracle = {self.state(munge(x), is_leaf=True): float(y)
          for x, y in zip(oracle_d['x'], oracle_d['y'])}
      
      # Scale rewards
      self.scaled_oracle = copy.copy(self.oracle)
      py = np.array(list(self.scaled_oracle.values()))

      # tfbind10 y has mean = 0, std = 0.26; has negative values.

      # expit normalization. R -> [0, 1]
      from scipy.special import expit
      py = expit(py * 3)

      REWARD_EXP = 3
      SCALE_REWARD_MAX = 10

      py = py ** REWARD_EXP
      py = SCALE_REWARD_MAX * py / max(py)
      self.scaled_oracle = {x: y for x, y in zip(self.scaled_oracle.keys(), py) if y > 0}

      # Rewards
      self.rs_all = [y for x, y in self.scaled_oracle.items()]
      assert all(r > 0 for r in self.rs_all)

      # Modes
      mode_percentile = 0.005
      self.mode_r_threshold = np.percentile(py, 100*(1-mode_percentile))

    # Core
    def reward(self, x):
      assert x.is_leaf, 'Error: Tried to compute reward on non-leaf node.'
      return self.scaled_oracle[x]

    def is_mode(self, x, r):
      return r >= self.mode_r_threshold

    '''
      Interpretation & visualization
    '''
    def dist_func(self, state1, state2):
      """ States are SeqPAState or SeqInsertState objects. """
      return levenshtein(state1.content, state2.content)

    def make_monitor(self):
      target = TargetRewardDistribution()
      target.init_from_base_rewards(self.rs_all)
      return Monitor(self.args, target, dist_func=self.dist_func,
                     is_mode_f=self.is_mode, callback=self.add_monitor)

    def add_monitor(self, xs, rs, allXtoR):
      """ Reimplement scoring with oracle, not unscaled oracle (used as R). """
      tolog = dict()
      return tolog

  return TFBind10MDP(args)


def main(args):
  print('Running experiment TFBind10 ...')

  if args.mdp_style == 'pa':
    base = seqpamdp.SeqPrependAppendMDP
    actorclass = seqpamdp.SeqPAActor
  elif args.mdp_style == 'insert':
    base = seqinsertmdp.SeqInsertMDP
    actorclass = seqinsertmdp.SeqInsertActor
  elif args.mdp_style == 'autoregressive':
    base = seqarmdp.SeqAutoregressiveMDP
    actorclass = seqarmdp.SeqARActor
  mdp = dynamic_inherit_mdp(base, args)

  actor = actorclass(args, mdp)
  model = models.make_model(args, mdp, actor)
  monitor = mdp.make_monitor()

  # Save memory, after constructing monitor with target rewards
  del mdp.rs_all

  trainer = trainers.Trainer(args, model, mdp, actor, monitor)
  trainer.learn()
  return
