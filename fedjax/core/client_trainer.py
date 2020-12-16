# Copyright 2020 Google LLC
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
"""Functions and interfaces for single and multi client training."""

import abc
import collections
import functools
import itertools

from typing import Any, Generic, Iterable, Iterator, List, NamedTuple, Optional, Tuple, TypeVar
from absl import flags

from fedjax.core import dataset_util
from fedjax.core import tree_util
from fedjax.core.model import Model
from fedjax.core.optimizer import Optimizer
from fedjax.core.typing import Batch
from fedjax.core.typing import FederatedData
from fedjax.core.typing import OptState
from fedjax.core.typing import Params
from fedjax.core.typing import PRNGKey
from fedjax.core.typing import PRNGSequence
from fedjax.core.typing import Updates
import jax
import jax.numpy as jnp
import jax.random as jrandom

T = TypeVar('T')

FLAGS = flags.FLAGS

flags.DEFINE_bool('use_parallel', True, '')


class ClientTrainer(Generic[T], metaclass=abc.ABCMeta):
  """Interface for training a single client.

  This interface defines the core training components that interact with trainer
  state. Here, trainer state is a simple container holding only the necessary
  values for training e.g. model parameters and optimizer state.

  NOTE: Trainer state is distinct from the federated algorithm state. However,
  trainer state is typically derived from the federated algorithm state.
  """

  @abc.abstractmethod
  def init_state(self, *args, **kwargs) -> T:
    """Initializes the client trainer state."""

  @abc.abstractmethod
  def one_step(self, client_trainer_state: T, batch: Batch, rng: PRNGKey) -> T:
    """Runs one step of training on input batch and updates the trainer state.

    Args:
      client_trainer_state: Current client trainer state.
      batch: Batch of example numpy arrays of shape [client_batch_size, ...].
      rng: PRNGKey.

    Returns:
      Updated trainer state.
    """

  def loop(self, init_client_trainer_state: T,
           examples: Iterable[Tuple[Batch, PRNGKey]]) -> Any:
    """Runs training loop over multiple batches.

    For most usecases, this default implementation should work.

    Args:
      init_client_trainer_state: Initial client trainer state.
      examples: Sequence of batches and rngs, where batches are numpy arrays of
        shape [client_batch_size, ...].

    Returns:
      Output client trainer state that is typically just an updated version of
        the input `init_client_trainer_state`. However, output is flexible.
    """
    client_trainer_state = init_client_trainer_state
    for batch, rng in examples:
      client_trainer_state = self.one_step(client_trainer_state, batch, rng)
    return client_trainer_state


def loop_step(trainer: ClientTrainer[T], init_client_trainer_state: T,
              batches: Iterable[Batch], rngs: Iterable[PRNGKey]) -> Any:

  # examples: Iterable[Tuple[Batch, PRNGKey]]) -> Any:
  return trainer.loop(init_client_trainer_state, zip(batches, rngs))


def split_batch(batch, new_size=None):
  return [
      collections.OrderedDict(
          {k: batch[k][i * new_size:(i + 1) * new_size]
           for k in batch.keys()})
      for i in range(int(batch[list(batch.keys())[0]].shape[0] / new_size))
  ]


def one_step_f(trainer: ClientTrainer[T], client_trainer_state: T, batch: Batch,
               rng: PRNGKey) -> Any:

  new_size = int(batch[list(batch.keys())[0]].shape[0] /
                 (dataset_util.MEGABATCH_MULTIPLIER))

  next_rng = rng
  for batch in split_batch(batch, new_size=new_size):
    rng, next_rng = jrandom.split(next_rng)
    client_trainer_state = trainer.one_step(client_trainer_state, batch, rng)
  return client_trainer_state, next_rng


def pmask(func, default_argnums=(0,)):
  """Wraps input function to add mask argument at first argument position.

  This is a convenience wrapper for masking functions that are intended to be
  run in parallel. Because most parallel functions in this module require that
  the leading axis dimension stay fixed, masking is necessary to support items
  that are less than the fixed size (i.e. remainder).

  NB: When using this function, remember that the argument indices will all be
  offset by +1 due to mask being set as the first argument.

  Args:
    func: Function that is intended to be run in parallel (e.g. train step).
    default_argnum: Argument index of default return value if mask is False.

  Returns:
    Wrapped mask function.
  """

  def masked_func(mask, *args, **kwargs):
    return jax.lax.cond(mask, lambda _: func(*args, **kwargs),
                        lambda _: tuple(args[i] for i in default_argnums), None)

  return masked_func


mask_step = pmask(one_step_f, default_argnums=(1, 3))
pmap_step = jax.pmap(mask_step, static_broadcasted_argnums=1)


class DefaultClientTrainerState(NamedTuple):
  """State container for `DefaultClientTrainer`.

  Attributes:
    params: Pytree of model parameters.
    opt_state: Pytree of optimizer state.
    weight: Accumulated scalar weight over all batches.
  """
  params: Params
  opt_state: OptState
  weight: float = 0.0


class DefaultClientTrainer(ClientTrainer):
  """Default single client trainer."""

  def __init__(self, model: Model, optimizer: Optimizer):
    super().__init__()
    self._model = model
    self._optimizer = optimizer

  def init_state(self,
                 params: Params,
                 opt_state: Optional[OptState] = None,
                 weight: float = 0.) -> DefaultClientTrainerState:
    if opt_state is None:
      opt_state = self._optimizer.init_fn(params)
    return DefaultClientTrainerState(
        params=params, opt_state=opt_state, weight=weight)

  @functools.partial(jax.jit, static_argnums=0)
  def one_step(self, client_trainer_state: DefaultClientTrainerState,
               batch: Batch, rng: PRNGKey) -> DefaultClientTrainerState:
    backward_pass_output = self._model.backward_pass(
        client_trainer_state.params, batch, rng)
    params_updates, opt_state = self._optimizer.update_fn(
        backward_pass_output.grads, client_trainer_state.opt_state)
    params = self._optimizer.apply_updates(client_trainer_state.params,
                                           params_updates)
    weight = client_trainer_state.weight + backward_pass_output.weight
    return DefaultClientTrainerState(
        params=params, opt_state=opt_state, weight=weight)


class ControlVariateTrainerState(NamedTuple):
  """State container for `ControlVariateTrainer`.

  Attributes:
    params: Pytree of model parameters.
    opt_state: Pytree of optimizer state.
    init_params: Pytree of initial model parameters.
    control_variate: Pytree of control variates that matches `params`.
    weight: Accumulated scalar weight over all batches.
  """
  params: Params
  opt_state: OptState
  init_params: Params
  control_variate: Updates
  weight: float = 0.


class ControlVariateTrainer(ClientTrainer):
  """Trainer with control variates."""

  def __init__(self, model: Model, base_optimizer: Optimizer):
    super().__init__()
    self._model = model
    self._base_optimizer = base_optimizer

  def init_state(self,
                 params: Params,
                 opt_state: OptState,
                 control_variate: Updates,
                 weight: float = 0.) -> ControlVariateTrainerState:
    return ControlVariateTrainerState(
        params=params,
        opt_state=opt_state,
        init_params=params,
        control_variate=control_variate,
        weight=weight)

  @functools.partial(jax.jit, static_argnums=0)
  def one_step(self, client_trainer_state: ControlVariateTrainerState,
               batch: Batch, rng: PRNGKey) -> ControlVariateTrainerState:
    client_control_variate = self._model.backward_pass(
        client_trainer_state.init_params, batch, rng).grads
    backward_pass_output = self._model.backward_pass(
        client_trainer_state.params, batch, rng)
    grads = backward_pass_output.grads
    adjusted_grads = jax.tree_multimap(lambda g, cc, c: g - cc + c, grads,
                                       client_control_variate,
                                       client_trainer_state.control_variate)
    updates, _ = self._base_optimizer.update_fn(adjusted_grads,
                                                client_trainer_state.opt_state)
    params = self._base_optimizer.apply_updates(client_trainer_state.params,
                                                updates)
    weight = client_trainer_state.weight + backward_pass_output.weight
    return ControlVariateTrainerState(
        params=params,
        opt_state=client_trainer_state.opt_state,
        init_params=client_trainer_state.init_params,
        control_variate=client_trainer_state.control_variate,
        weight=weight)


def train_single_client(dataset: dataset_util.DatasetOrIterable,
                        client_trainer: ClientTrainer[T],
                        init_client_trainer_state: T,
                        rng_seq: PRNGSequence) -> Any:
  """Trains model for a single client's dataset.

  Args:
    dataset: Dataset of examples [client_batch_size, d].
    client_trainer: ClientTrainer instance.
    init_client_trainer_state: Initial client trainer state. This will typically
      be derived from the federated algorithm state before calling
      `train_multiple_clients`.
    rng_seq: Random key generator.

  Returns:
    Output of client trainer that is typically just an updated version of the
      input `init_client_trainer_state`. However, output is flexible.
  """
  examples = zip(dataset_util.iterate(dataset), rng_seq)
  client_trainer_state = client_trainer.loop(init_client_trainer_state,
                                             examples)
  return client_trainer_state


def train_multiple_clients(
    federated_data: FederatedData, client_ids: List[str],
    client_trainer: ClientTrainer[T], init_client_trainer_state: T,
    rng_seq: PRNGSequence,
    client_data_hparams: dataset_util.ClientDataHParams) -> Iterator[Any]:
  """Trains separate model for each client and records client updates.

  Args:
    federated_data: Federated data separated per client.
    client_ids: Ids of clients to train.
    client_trainer: ClientTrainer instance.
    init_client_trainer_state: Initial client trainer state. This will typically
      be derived from algorithm state before calling `train_multiple_clients`.
    rng_seq: Random key generator.
    client_data_hparams: Hyperparameters for client dataset preparation.

  Yields:
    Output of client trainer that is typically just an updated version of the
      input `init_client_trainer_state`. However, output is flexible.
  """
  for client_id in client_ids:
    client_dataset = federated_data.create_tf_dataset_for_client(client_id)
    client_dataset = dataset_util.preprocess_tf_dataset(client_dataset,
                                                        client_data_hparams)
    examples = zip(client_dataset.as_numpy_iterator(), rng_seq)
    client_trainer_state = client_trainer.loop(init_client_trainer_state,
                                               examples)
    yield client_trainer_state


def empty_iterator():
  yield from ()


def peek(iterator):
  sentinel = object()
  if isinstance(iterator, list):
    item = sentinel if not iterator else iterator[0]
  else:
    item = next(iterator, sentinel)
  if item is not sentinel:
    return item, itertools.chain([item], iterator)


def peek_fillvalue(iterators):
  """Peeks into iterators items to create a fillvalue of the same structure."""
  sentinel = object()
  item = sentinel
  for i, iterator in enumerate(iterators):
    if not iterator:
      continue
    peek_output = peek(iterator)
    if peek_output is not None:
      item, peek_iterator = peek_output
      iterators[i] = peek_iterator
      break
  if item is sentinel:
    raise ValueError('All iterators fully consumed.')
  return jax.tree_map(jnp.zeros_like, item)


def train_multiple_clients_parallel(
    federated_data: FederatedData, client_ids: List[str],
    client_trainer: ClientTrainer[T], init_client_trainer_state: T,
    rng_seq: PRNGSequence,
    client_data_hparams: dataset_util.ClientDataHParams) -> Iterator[Any]:
  """Trains separate model for each client and records client updates.

  It has the same inputs and return values as `train_multiple_clients` above,
  but parallelizes the training across multiple devices if available.

  Args:
    federated_data: Federated data separated per client.
    client_ids: Ids of clients to train.
    client_trainer: ClientTrainer instance.
    init_client_trainer_state: Initial client trainer state. This will typically
      be derived from algorithm state before calling `train_multiple_clients`.
    rng_seq: Random key generator.
    client_data_hparams: Hyperparameters for client dataset preparation.

  Yields:
    Output of client trainer that is typically just an updated version of the
      input `init_client_trainer_state`. However, output is flexible.
  """

  def _create_examples(client_id):
    client_dataset = federated_data.create_tf_dataset_for_client(client_id)
    client_dataset = dataset_util.preprocess_tf_dataset(client_dataset,
                                                        client_data_hparams)

    return client_dataset.as_numpy_iterator()

  client_data_hparams = dataset_util.ClientDataHParams(
      batch_size=client_data_hparams.batch_size,
      num_epochs=client_data_hparams.num_epochs,
      drop_remainder=FLAGS.use_parallel or client_data_hparams.drop_remainder,
      shuffle_buffer_size=client_data_hparams.shuffle_buffer_size,
      num_batches=client_data_hparams.num_batches)

  num_devices = jax.device_count()

  init_stack_state = tree_util.tree_broadcast(init_client_trainer_state,
                                              num_devices)
  stack_rng = jrandom.split(next(rng_seq), num_devices)

  client_data = [_create_examples(client_id) for client_id in client_ids]
  fillvalue = peek_fillvalue(client_data)

  client_data = list(
      tuple(
          zip(*sorted(
              map(lambda batches: (len(batches), batches),
                  ([batch
                    for i, batch in enumerate(stream)]
                   for stream in client_data)),
              key=lambda k: k[0])))[1])

  # (TODO) Handle number of clients not divisible by num_devices.
  for i in range(int(len(client_ids) / num_devices) + 1):
    stack_state = init_stack_state
    # next_clients = client_ids[num_devices * i: num_devices * (i + 1)]
    # streams = [_create_examples(client_id) for client_id in next_clients]

    streams = client_data[num_devices * i:num_devices * (i + 1)]
    if not streams:
      break

    if len(streams) < num_devices:
      client_count = len(streams)
      streams += [[fillvalue] for _ in range(num_devices - len(streams))]
    else:
      client_count = num_devices

    for batches in itertools.zip_longest(*streams, fillvalue=fillvalue):

      # rng = [next(rng_seq) for _ in batches]
      mask = jnp.array([b is not fillvalue for b in batches])
      # stack_rng = tree_util.tree_stack(rng)
      stack_mask = tree_util.tree_stack(mask)
      stack_batch = tree_util.tree_stack(batches)
      # Mask must be the first input.
      stack_state, stack_rng = pmap_step(stack_mask, client_trainer,
                                         stack_state, stack_batch, stack_rng)

    # Unstack stack_state, yield each one.
    for j, final_state in enumerate(tree_util.tree_unstack(stack_state)):
      if j < client_count:
        yield final_state
