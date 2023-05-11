from itertools import chain
from tqdm import tqdm
import numpy as np
import torch
import wandb

from .basegfn import BaseTBGFlowNet, tensor_to_np


class Empty(BaseTBGFlowNet):
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
  
  def train(self, batch):
    return


class TBGFN(BaseTBGFlowNet):
  """ Trajectory balance GFN. Learns forward and backward policy. """
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
    print('Model: TBGFN')

  def train(self, batch):
    return self.train_tb(batch)


class MaxEntGFN(BaseTBGFlowNet):
  """ Maximum Entropy GFlowNet with fixed uniform backward policy. 

      Methods back_logps_unique, back_sample override parent BaseTBGFlowNet
      methods, which simply call the backward policy's functions.    
  """
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
    print('Model: MaxEntGFN')

  def train(self, batch):
    return self.train_tb(batch)

  def back_logps_unique(self, batch):
    """ Uniform distribution over parents.

        Other idea - just call parent back_logps_unique, then replace
        predicted logps.
        see policy.py : logps_unique(batch)

        Output logps of unique children/parents.

        Typical logic flow (example for getting children)
        1. Call network on state - returns high-dim actions
        2. Translate actions into list of states - not unique
        3. Filter invalid child states
        4. Reduce states to unique, using hash property of states.
           Need to add predicted probabilities.
        5. Normalize probs to sum to 1

        Input: List of [State], n items
        Returns
        -------
        logps: n-length List of torch.tensor of logp.
            Each tensor can have different length.
        states: List of List of [State]; must be unique.
            Each list can have different length.
    """
    batched = bool(type(batch) is list)
    if not batched:
      batch = [batch]

    batch_dicts = []
    for state in batch:
      parents = self.mdp.get_unique_parents(state)
      logps = np.log([1/len(parents) for parent in parents])

      state_to_logp = {parent: logp for parent, logp in zip(parents, logps)}
      batch_dicts.append(state_to_logp)
    return batch_dicts if batched else batch_dicts[0]

  def back_sample(self, batch):
    """ Uniformly samples a parent.

        Typical logic flow skips some steps in logps_unique.
        1. Call network on state - return high-dim actions
        2. Translate actions into list of states - not unique
        3. Filter invalid child states
        4. Skipped - no need to reduce states to unique.
        5. Normalize probs to sum to 1
        Return sample

        Input: batch, List of [State]
        Output: List of [State]
    """
    batched = bool(type(batch) is list)
    if not batched:
      batch = [batch]

    batch_samples = []
    for state in batch:
      sample = np.random.choice(self.mdp.get_unique_parents(state))
      batch_samples.append(sample)
    return batch_samples if batched else batch_samples[0]


class SubstructureGFN(BaseTBGFlowNet):
  """ Substructure GFN. Learns with guided trajectory balance. """
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
    print('Model: Substructure GFN')

  def train(self, batch):
    return self.train_substructure(batch)

  def train_substructure(self, batch, log = True):
    """ Guided trajectory balance for substructure GFN.
        1. Update back policy to approximate guide,
        2. Update forward policy to match back policy with TB.
        
        Batch: List of [Experience]

        Uses 1 pass for fwd and back net.
    """
    fwd_chain = self.batch_traj_fwd_logp(batch)
    back_chain = self.batch_traj_back_logp(batch)

    # 1. Obtain back policy loss
    logp_guide = torch.stack([exp.logp_guide for exp in batch])
    back_losses = torch.square(back_chain - logp_guide)
    back_losses = torch.clamp(back_losses, max=10**2)
    mean_back_loss = torch.mean(back_losses)

    # 2. Obtain TB loss with target: mix back_chain with logp_guide
    targets = []
    for i, exp in enumerate(batch):
      if exp.logp_guide is not None:
        w = self.args.target_mix_backpolicy_weight
        target = w * back_chain[i].detach() + (1 - w) * (exp.logp_guide + exp.logr)
      else:
        target = back_chain[i].detach()
      targets.append(target)
    targets = torch.stack(targets)

    tb_losses = torch.square(fwd_chain - targets)
    tb_losses = torch.clamp(tb_losses, max=10**2)
    loss_tb = torch.mean(tb_losses)

    # 1. Update back policy on back loss
    self.optimizer_back.zero_grad()
    loss_step1 = mean_back_loss
    loss_step1.backward()
    for param_set in self.clip_grad_norm_params:
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm, error_if_nonfinite=True)
    self.optimizer_back.step()
    if log:
      loss_step1 = tensor_to_np(loss_step1)
      print(f'Back training:', loss_step1)

    # 2. Update fwd policy on TB loss
    self.optimizer_fwdZ.zero_grad()
    loss_tb.backward()
    for param_set in self.clip_grad_norm_params:
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm, error_if_nonfinite=True)
    self.optimizer_fwdZ.step()
    self.clamp_logZ()
    if log:
      loss_tb = tensor_to_np(loss_tb)
      print(f'Fwd training:', loss_tb)

    if log:
      logZ = tensor_to_np(self.logZ)
      print(f'{logZ=}')
      wandb.log({
        'Sub back loss': loss_step1,
        'Sub fwdZ loss': loss_tb,
        'Sub logZ': logZ,
      })
    return


def make_model(args, mdp, actor):
  """ Constructs MaxEnt / TB / Sub GFN. """
  if args.model == 'maxent':
    model = MaxEntGFN(args, mdp, actor)
  elif args.model == 'tb':
    model = TBGFN(args, mdp, actor)
  elif args.model == 'sub':
    model = SubstructureGFN(args, mdp, actor)
  elif args.model == 'random':
    args.explore_epsilon = 1.0
    args.num_offline_batches_per_round = 0
    model = Empty(args, mdp, actor)
  return model