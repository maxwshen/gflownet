"""
  seh as string
"""
import pickle, functools
import numpy as np

import gflownet.trainers as trainers
from gflownet.MDPs import molstrmdp
from gflownet.monitor import TargetRewardDistribution, Monitor
from gflownet.GFNs import models

from datasets.sehstr import gbr_proxy

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import FingerprintSimilarity


class SEHstringMDP(molstrmdp.MolStrMDP):
  def __init__(self, args):
    super().__init__(args)
    self.args = args

    assert args.blocks_file == 'datasets/sehstr/block_18.json', 'ERROR - x_to_r and rewards are designed for block_18.json'

    self.proxy_model = gbr_proxy.sEH_GBR_Proxy(args)

    with open('datasets/sehstr/sehstr_gbtr_allpreds.pkl', 'rb') as f:
      self.rewards = pickle.load(f)

    # scale rewards
    py = np.array(list(self.rewards))

    self.SCALE_REWARD_MAX = 10
    self.SCALE_MIN = 1e-3
    self.REWARD_EXP = 6

    py = np.maximum(py, self.SCALE_MIN)
    py = py ** self.REWARD_EXP
    self.scale = self.SCALE_REWARD_MAX / max(py)
    py = py * self.scale

    self.scaled_rewards = py

    # define modes as top % of xhashes.
    mode_percentile = 0.001
    self.mode_r_threshold = np.percentile(py, 100*(1-mode_percentile))

  # Core
  @functools.cache
  def reward(self, x):
    assert x.is_leaf, 'Error: Tried to compute reward on non-leaf node.'
    pred = self.proxy_model.predict_state(x)
    r = np.maximum(pred, self.SCALE_MIN)
    r = r ** self.REWARD_EXP
    r = r * self.scale
    return r

  def is_mode(self, x, r):
    return r >= self.mode_r_threshold

  # Diversity
  def dist_states(self, state1, state2):
    """ Tanimoto similarity on morgan fingerprints """
    fp1 = self.get_morgan_fp(state1)
    fp2 = self.get_morgan_fp(state2)
    return 1 - FingerprintSimilarity(fp1, fp2)

  @functools.cache
  def get_morgan_fp(self, state):
    mol = self.state_to_mol(state)
    fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    return fp

  """
    Interpretation & visualization
  """
  def make_monitor(self):
    """ Make monitor, called during training. """
    target = TargetRewardDistribution()
    target.init_from_base_rewards(self.scaled_rewards)
    return Monitor(self.args, target, dist_func=self.dist_states,
                   is_mode_f=self.is_mode)

  def reduce_storage(self):
    del self.rewards
    del self.scaled_rewards


def main(args):
  print('Running experiment sehstr ...')
  mdp = SEHstringMDP(args)
  actor = molstrmdp.MolStrActor(args, mdp)
  model = models.make_model(args, mdp, actor)
  monitor = mdp.make_monitor()

  mdp.reduce_storage()

  trainer = trainers.Trainer(args, model, mdp, actor, monitor)
  trainer.learn()
  return
