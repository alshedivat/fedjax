# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Federated posterior averaging implementation.

Based on the paper:

Federated Learning via Posterior Averaging: A New Perspective and Practical Algorithms
    Maruan Al-Shedivat, Jennifer Gillenwater, Eric Xing, Afshin Rostamizadeh.
    ICLR 2021. https://arxiv.org/abs/2010.05273
"""

from collections import defaultdict
import functools
import math
from typing import Iterable, List, NamedTuple, Tuple

from fedjax import core
import jax
from jax import random
import jax.numpy as jnp
import numpyro.infer as infer


class MCMCHParams(NamedTuple):
  """Hyperparameters for MCMC sampling.

  Attributes:
    kerenel: The MCMC kernel used by the sampler.
    num_warmup: The number of warmup samples.
    num_samples: The number of samples to draw.
    num_chains: The numbner of MCMC chains to run in parallel.
    step_size: The step size used by HMC and NUTS.
    trajectory_length: The trajectory length used by HMC.
  """
  kernel: infer.mcmc.MCMCKernel
  num_warmup: int
  num_samples: int
  num_chains: int = 1
  step_size: float = 1.0
  trajectory_length: float = 2 * math.pi


class FedPAHParams(NamedTuple):
  """Hyperparameters for federated posterior averaging.

  Attributes:
    train_data_hparams: Hyperparameters for training client data preparation.
    mcmc_hparams: Hyperparameters for MCMC sampling.
    rho: The shrinkage parameter.
  """
  train_data_hparams: core.ClientDataHParams
  mcmc_hparams: MCMCHParams
  rho: float


class FedPAState(NamedTuple):
  """The server state for FedPA passed between rounds.

  Attributes:
    params: A pytree representing the server model parameters.
    server_opt_state: A pytree representing the base optimizer state.
  """
  params: core.Params
  server_opt_state: core.OptState


class FedPAClientTrainerState(NamedTuple):
  """The state of the FedPA client trainer.

  Attributes:
    params: Pytree of model parameters.
    num_examples: The size of the client dataset.
  """
  params: core.Params
  num_examples: float = 0.


class FedPAClientTrainer(core.ClientTrainer):
  """Single client trainer that runs local posterior inference."""

  def __init__(self,
               model: core.Model,
               mcmc_hparams: MCMCHParams,
               rho: float):
    super().__init__()
    self._model = model
    self._mcmc_hparams = mcmc_hparams
    self._rho = rho

  def init_state(self,
                 params: core.Params,
                 num_examples: float = 0.) -> FedPAClientTrainerState:
    return FedPAClientTrainerState(params=params, num_examples=num_examples)

  def compute_delta(self,
                    init_params: core.Params,
                    samples: core.Params) -> core.Params:
    """Computes client delta using dynamic programming.

    Args:
      init_params: Initial model parameters.
      samples: (Approximate) posterior samples.

    Returns:
      Updated model parameters.
    """
    dp = defaultdict(list)

    # Initialize running average and delta.
    num_samples = self._mcmc_hparams.num_samples
    samples_ra = core.tree_map(lambda x: x[0], samples)
    delta = core.tree_multimap(lambda a, b: a - b, init_params, samples_ra)

    for t, i in enumerate(range(1, num_samples), 2):
      # Select a new sample.
      s = core.tree_map(lambda x: x[i], samples)
      # Compute initial u and v.
      u = v = core.tree_multimap(lambda a, b: a - b, s, samples_ra)
      # Compute v_{t-1,t} (solution of `sigma_{t-1} x = u_t`).
      for k, (v_k, dot_uk_vk) in enumerate(zip(dp["v"], dp["dot_u_v"]), 2):
        g_k = self._rho * (k - 1) / k
        v = core.tree_multimap(
          lambda a, b, c, d: b - g_k * jnp.sum(a * c) / (1 + g_k * d) * c,
          u, v, v_k, dot_uk_vk)
      # Compute `dot(u_t, v_t)` and `dot(u_t, delta_t)`.
      dot_u_v = core.tree_multimap(lambda a, b: jnp.sum(a * b), u, v)
      dot_u_d = core.tree_multimap(lambda a, b: jnp.sum(a * b), u, delta)
      # Compute delta.
      gamma = self._rho * (t - 1) / t
      coeff = core.tree_multimap(
        lambda a, b: gamma * (t * a - b) / (1 + gamma * b),
        dot_u_d, dot_u_v)
      delta = core.tree_multimap(
        lambda a, b, c: a - (1 + c) * b / t,
        delta, v, coeff)
      # Update the DP state.
      dp["v"].append(v)
      dp["dot_u_v"].append(dot_u_v)
      # Update running mean of the samples.
      samples_ra = core.tree_multimap(lambda a, b: ((t - 1) * a + b) / t,
                                      samples_ra, s)

    return core.tree_map(lambda x: x * (1 + (num_samples - 1) * self._rho),
                         delta)

  # @functools.partial(jax.jit, static_argnums=0)
  def one_step(self,
               client_trainer_state: FedPAClientTrainerState,
               batches: List[core.Batch],
               rng: core.PRNGKey) -> core.Params:
    init_params = client_trainer_state.params
    rng_model, rng_mcmc = random.split(rng)

    # Define the potential function proportional to the local loss.
    def potential_fn(params: core.Params):
      loss = 0.
      for batch in batches:
        preds = self._model.apply_fn(params, rng_model, batch,
                                     **self._model.train_kwargs)
        loss_metric = self._model.loss_fn(batch, preds)
        loss += loss_metric.result()
      return loss / len(batches) + self._model.reg_fn(params)

    # Produce local posterior samples.
    mcmc = infer.MCMC(
      self._mcmc_hparams.kernel(
        potential_fn=potential_fn,
        step_size=self._mcmc_hparams.step_size,
        trajectory_length=self._mcmc_hparams.trajectory_length),
      num_warmup=self._mcmc_hparams.num_warmup,
      num_samples=self._mcmc_hparams.num_samples,
      num_chains=self._mcmc_hparams.num_chains,
      progress_bar=True)
    mcmc.run(rng_mcmc, init_params=init_params)
    samples = mcmc.get_samples()

    # Compute delta.
    return self.compute_delta(init_params, samples)

  def loop(self, init_client_trainer_state: FedPAClientTrainerState,
           examples: Iterable[Tuple[core.Batch, core.PRNGKey]]) -> FedPAClientTrainerState:
    """Loops over the local dataset, then runs MCMC and computes updated state.

    Args:
      init_client_trainer_state: Initial client trainer state.
      examples: Sequence of batches and rngs, where batches are numpy arrays of
        shape [client_batch_size, ...].

    Returns:
      An updated version of the input `init_client_trainer_state`.
    """
    num_examples = 0.
    batches, last_rng = [], None
    for batch, rng in examples:
      num_examples += batch["x"].shape[0]
      batches.append(batch)
      last_rng = rng
    delta_params = self.one_step(init_client_trainer_state, batches, last_rng)
    return FedPAClientTrainerState(params=delta_params,
                                   num_examples=num_examples)


class FedPA(core.FederatedAlgorithm):
  """Federated posterior averaging algorithm."""

  def __init__(self,
               federated_data: core.FederatedData,
               model: core.Model,
               server_optimizer: core.Optimizer,
               hparams: FedPAHParams,
               rng_seq: core.PRNGSequence):
    """Initializes FedAvg algorithm.

    Args:
      federated_data: Federated data separated per client.
      model: Model implementation.
      server_optimizer: Server optimizer.
      client_mcmc_kernel: Client MCMC kernel.
      hparams: Hyperparameters for federated posterior averaging.
      rng_seq: Iterator of JAX random keys.
    """
    self._federated_data = federated_data
    self._model = model
    self._server_optimizer = server_optimizer
    self._client_data_hparams = hparams.train_data_hparams
    self._client_trainer = FedPAClientTrainer(model=model,
                                              mcmc_hparams=hparams.mcmc_hparams,
                                              rho=hparams.rho)
    self._rng_seq = rng_seq

  @property
  def federated_data(self) -> core.FederatedData:
    return self._federated_data

  @property
  def model(self) -> core.Model:
    return self._model

  def init_state(self) -> FedPAState:
    params = self._model.init_params(next(self._rng_seq))
    server_opt_state = self._server_optimizer.init_fn(params)
    return FedPAState(params=params, server_opt_state=server_opt_state)

  def run_round(self, state: FedPAState, client_ids: List[str]) -> FedPAState:
    """Runs one round of federated averaging."""
    # Train model per client.
    client_updates = core.train_multiple_clients(
      federated_data=self.federated_data,
      client_ids=client_ids,
      client_trainer=self._client_trainer,
      init_client_trainer_state=self._client_trainer.init_state(state.params),
      client_data_hparams=self._client_data_hparams,
      rng_seq=self._rng_seq)

    # Weighted average of param delta across clients.
    delta_params_and_weight = map(lambda x: (x.params, x.num_examples),
                                  client_updates)
    delta_params = core.tree_mean(delta_params_and_weight)

    # Server state update.
    updates, server_opt_state = self._server_optimizer.update_fn(
        delta_params, state.server_opt_state)
    params = self._server_optimizer.apply_updates(state.params, updates)
    return FedPAState(params=params, server_opt_state=server_opt_state)
