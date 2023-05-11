"""
  qm9 as string
"""
import pickle, functools
import numpy as np

import gflownet.trainers as trainers
from gflownet.MDPs import molstrmdp
from gflownet.monitor import TargetRewardDistribution, Monitor
from gflownet.GFNs import models

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import FingerprintSimilarity


class QM9stringMDP(molstrmdp.MolStrMDP):
  def __init__(self, args):
    super().__init__(args)
    self.args = args

    x_to_r_file = args.x_to_r_file

    # Read from file
    print(f'Loading data ...')
    with open(x_to_r_file, 'rb') as f:
      self.oracle = pickle.load(f)
    
    # scale rewards
    py = np.array(list(self.oracle.values()))

    SCALE_REWARD_MAX = 100
    SCALE_MIN = 1e-3
    REWARD_EXP = 5

    py = np.maximum(py, SCALE_MIN)
    py = py ** REWARD_EXP
    py = py * (SCALE_REWARD_MAX / max(py))

    self.scaled_oracle = {x: y for x, y in zip(self.oracle.keys(), py) if y > 0}
    assert min(self.scaled_oracle.values()) > 0

    # define modes as top % of xhashes.
    mode_percentile = 0.005
    num_modes = int(len(self.scaled_oracle) * mode_percentile)
    sorted_xs = sorted(self.scaled_oracle, key=self.scaled_oracle.get)
    self.modes = sorted_xs[-num_modes:]
    print(f'Found {len(self.modes)=}')

  # Core
  def reward(self, x):
    assert x.is_leaf, 'Error: Tried to compute reward on non-leaf node.'
    return self.scaled_oracle[x.content]

  def is_mode(self, x, r):
    return x.content in self.modes

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
    rs_all = list(self.scaled_oracle.values())
    target.init_from_base_rewards(rs_all)
    return Monitor(self.args, target, dist_func=self.dist_states,
                   is_mode_f=self.is_mode)


def main(args):
  print('Running experiment qm9str ...')
  mdp = QM9stringMDP(args)
  actor = molstrmdp.MolStrActor(args, mdp)
  model = models.make_model(args, mdp, actor)
  monitor = mdp.make_monitor()
  trainer = trainers.Trainer(args, model, mdp, actor, monitor)
  trainer.learn()
  return
