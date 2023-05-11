import random, time
import numpy as np
import torch
import wandb
from tqdm import tqdm
import ray

from . import guide
from .data import Experience


class Trainer:
  def __init__(self, args, model, mdp, actor, monitor):
    self.args = args
    self.model = model
    self.mdp = mdp
    self.actor = actor
    self.monitor = monitor

  def learn(self, *args, **kwargs):
    if self.args.model == 'sub':
      print(f'Learning with ray guide workers ...')
      self.learn_with_ray_workers(*args, **kwargs)
    else:
      print(f'Learning without guide workers ...')
      self.learn_default(*args, **kwargs)

  def handle_init_dataset(self, initial_XtoR):
    if initial_XtoR:
      print(f'Using initial dataset of size {len(initial_XtoR)}. \
              Skipping first online round ...')
      if self.args.init_logz:
        self.model.init_logz(np.log(sum(initial_XtoR.values())))
    else:
      print(f'No initial dataset used')
    return

  """
    Training
  """
  def learn_default(self, initial_XtoR=None):
    """ Main learning training loop.
        Each learning round:
          Each online batch:
            sample a new dataset using exploration policy.
          Each offline batch:
            resample batch from full historical dataset
        Monitor exploration - judge modes with monitor_explore callable.

        To learn on fixed dataset only: Set 0 online batches per round,
        and provide initial dataset.

        dataset = List of [Experience]
    """
    allXtoR = initial_XtoR if initial_XtoR else dict()
    self.handle_init_dataset(initial_XtoR)

    num_online = self.args.num_online_batches_per_round
    num_offline = self.args.num_offline_batches_per_round
    online_bsize = self.args.num_samples_per_online_batch
    offline_bsize = self.args.num_samples_per_offline_batch
    monitor_fast_every = self.args.monitor_fast_every
    monitor_num_samples = self.args.monitor_num_samples
    print(f'Starting active learning. \
            Each round: {num_online=}, {num_offline=}')

    for round_num in tqdm(range(self.args.num_active_learning_rounds)):
      print(f'Starting learning round {round_num+1} / {self.args.num_active_learning_rounds} ...')
      
      # Online training - skip first if initial dataset was provided
      if not initial_XtoR or round_num > 0:
        for _ in range(num_online):
          # Sample new dataset
          with torch.no_grad():
            explore_data = self.model.batch_fwd_sample(online_bsize,
                epsilon=self.args.explore_epsilon)

          # Save to full dataset
          for exp in explore_data:
            if exp.x not in allXtoR:
              allXtoR[exp.x] = exp.r          

          # Train on online dataset
          for step_num in range(self.args.num_steps_per_batch):
            self.model.train(explore_data)

      # Offline training
      for _ in range(num_offline):
        offline_xs = self.select_offline_xs(allXtoR, offline_bsize)
        offline_dataset = self.offline_PB_traj_sample(offline_xs, allXtoR)

        # Train
        for step_num in range(self.args.num_steps_per_batch):
          self.model.train(offline_dataset)

      if round_num % monitor_fast_every == 0 and round_num > 0:
        truepolicy_data = self.model.batch_fwd_sample(monitor_num_samples,
              epsilon=0)
        self.monitor.log_samples(round_num, truepolicy_data)

      self.monitor.maybe_eval_samplelog(self.model, round_num, allXtoR)

      if round_num % self.args.save_every_x_active_rounds == 0:
        if round_num > 0:
          self.model.save_params(self.args.saved_models_dir + \
                                 self.args.run_name + \
                                 f'_round_{round_num}.pth')

    print('Finished training.')
    self.model.save_params(self.args.saved_models_dir + \
                           self.args.run_name + '_final.pth')
    self.monitor.maybe_eval_samplelog(self.model, round_num, allXtoR)
    return

  """
    Learn with ray workers
  """
  def learn_with_ray_workers(self, initial_XtoR=None):
    """ Guided trajectory balance - ray workers compute guide trajectories.
    """
    # Ray init
    # ray.init(num_cpus = self.args.num_guide_workers)
    guidemanager = guide.RayManager(self.args, self.mdp)

    allXtoR = initial_XtoR if initial_XtoR else dict()
    guidemanager.update_allXtoR(allXtoR)
    self.handle_init_dataset(initial_XtoR)

    num_online = self.args.num_online_batches_per_round
    num_offline = self.args.num_offline_batches_per_round
    online_bsize = self.args.num_samples_per_online_batch
    offline_bsize = self.args.num_samples_per_offline_batch
    monitor_fast_every = self.args.monitor_fast_every
    monitor_num_samples = self.args.monitor_num_samples
    print(f'Starting active learning. \
            Each round: {num_online=}, {num_offline=}')

    for round_num in tqdm(range(self.args.num_active_learning_rounds)):
      print(f'Starting learning round {round_num+1}/{self.args.num_active_learning_rounds} ...')
      
      # 1. Sample online x with explore policy
      for _ in range(num_online):
        print(f'Sampling new x ...')
        with torch.no_grad():
          explore_data = self.model.batch_fwd_sample(online_bsize,
              epsilon=self.args.explore_epsilon)

        online_xs = [exp.x for exp in explore_data]
        for exp in explore_data:
          if exp.x not in allXtoR:
            allXtoR[exp.x] = exp.r
        guidemanager.update_allXtoR(allXtoR)

        # 2. Submit online jobs - get guide traj for x
        guidemanager.submit_online_jobs(online_xs)

      # 2a. Submit offline jobs
      for _ in range(num_offline):
        offline_xs = self.select_offline_xs(allXtoR, offline_bsize)

        # Submit offline jobs
        if self.args.offline_style == 'guide_resamples_traj':
          guidemanager.submit_online_jobs(offline_xs)
        if self.args.offline_style == 'guide_scores_back_policy_traj':
          print(f'Sampling offline trajectories with back policy ...')
          with torch.no_grad():
            offline_trajs = self.model.batch_back_sample(offline_xs)
          guidemanager.submit_offline_jobs(offline_trajs)

      # 4. Train if possible
      for _ in range(num_online + num_offline):
        batch = guidemanager.get_results(batch_size=online_bsize)
        if batch is not None:
          print(f'Training ...')
          for step_num in range(self.args.num_steps_per_batch):
            self.model.train(batch)

      # 5. End of active round - monitor and save
      if round_num % monitor_fast_every == 0 and round_num > 0:
        truepolicy_data = self.model.batch_fwd_sample(monitor_num_samples,
              epsilon=0)
        self.monitor.log_samples(round_num, truepolicy_data)

      # Save to full dataset & log to monitor
      self.monitor.maybe_eval_samplelog(self.model, round_num, allXtoR)

      if round_num and round_num % self.args.save_every_x_active_rounds == 0:
        self.model.save_params(self.args.saved_models_dir + \
                               self.args.run_name + f'_round_{round_num}.pth')

    print('Finished training.')
    self.model.save_params(self.args.saved_models_dir + \
                           self.args.run_name + '_final.pth')
    self.monitor.maybe_eval_samplelog(self.model, round_num, allXtoR)
    return

  """
    Offline training
  """
  def select_offline_xs(self, allXtoR, batch_size):
    select = self.args.get('offline_select', 'biased')
    if select == 'biased':
      return self.__biased_sample_xs(allXtoR, batch_size)
    elif select == 'random':
      return self.__random_sample_xs(allXtoR, batch_size)

  def __biased_sample_xs(self, allXtoR, batch_size):
    """ Select xs for offline training. Returns List of [State].
        Draws 50% from top 10% of rewards, and 50% from bottom 90%. 
    """
    if len(allXtoR) < 10:
      return []
    rewards = np.array(list(allXtoR.values()))
    threshold = np.percentile(rewards, 90)
    top_xs = [x for x, r in allXtoR.items() if r >= threshold]
    bottom_xs = [x for x, r in allXtoR.items() if r <= threshold]
    sampled_xs = random.choices(top_xs, k=batch_size // 2) + \
                 random.choices(bottom_xs, k=batch_size // 2)
    return sampled_xs

  def __random_sample_xs(self, allXtoR, batch_size):
    """ Select xs for offline training. Returns List of [State]. """
    return random.choices(list(allXtoR.keys()), k=batch_size)

  def offline_PB_traj_sample(self, offline_xs, allXtoR):
    """ Sample trajectories for x using P_B, for offline training with TB.
        Returns List of [Experience].
    """
    offline_rs = [allXtoR[x] for x in offline_xs]

    # Not subgfn: sample trajectories from backward policy
    print(f'Sampling trajectories from backward policy ...')
    with torch.no_grad():
      offline_trajs = self.model.batch_back_sample(offline_xs)

    offline_dataset = [
      Experience(traj=traj, x=x, r=r,
                  logr=torch.log(torch.tensor(r, device=self.args.device)))
      for traj, x, r in zip(offline_trajs, offline_xs, offline_rs)
    ]
    return offline_dataset

