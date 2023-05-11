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

  class TFBind8MDP(base):
    def __init__(self, args):
      super().__init__(args,
                       alphabet=list('0123'),
                       forced_stop_len=8)
      self.args = args

      # Read from file
      print(f'Loading data ...')
      with open('datasets/tfbind8/tfbind8-exact-v0-all.pkl', 'rb') as f:
        oracle_d = pickle.load(f)
      
      munge = lambda x: ''.join([str(c) for c in list(x)])
      self.oracle = {self.state(munge(x), is_leaf=True): float(y)
          for x, y in zip(oracle_d['x'], oracle_d['y'])}
      
      # Scale rewards
      self.scaled_oracle = copy.copy(self.oracle)
      py = np.array(list(self.scaled_oracle.values()))

      REWARD_EXP = 3
      py = py ** REWARD_EXP

      py = py * (args.scale_reward_max / max(py))
      py = np.maximum(py, 0.001)
      self.scaled_oracle = {x: y for x, y in zip(self.scaled_oracle.keys(), py)}

      # Rewards
      self.rs_all = [y for x, y in self.scaled_oracle.items()]

      # Modes
      with open('datasets/tfbind8/modes_tfbind8.pkl', 'rb') as f:
        modes = pickle.load(f)
      self.modes = set([self.state(munge(x), is_leaf=True) for x in modes])

    # Core
    def reward(self, x):
      assert x.is_leaf, 'Error: Tried to compute reward on non-leaf node.'
      return self.scaled_oracle[x]

    def is_mode(self, x, r):
      return x in self.modes

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

  return TFBind8MDP(args)


def main(args):
  print('Running experiment TFBind8 ...')

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
