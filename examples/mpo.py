# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""An MPO agent trained on BSuite's Catch environment."""

import collections
import functools
import random
from absl import app
from absl import flags
from bsuite.environments import catch, cartpole, mountain_car
import haiku as hk
from haiku import nets
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax

# Allow running as `python -m examples.mpo` or `python examples/mpo.py`.
try:  # pragma: no cover - import shim
    from examples import experiment
except ImportError:  # pragma: no cover
    import importlib
    import pathlib
    import sys

    sys.path.append(str(pathlib.Path(__file__).resolve().parent))
    experiment = importlib.import_module("experiment")

# bsuite<=0.3 uses deprecated np.int; restore alias for NumPy>=1.20.
if not hasattr(np, "int"):  # pragma: no cover
    np.int = int

Batch = collections.namedtuple("Batch", "obs_tm1 a_tm1 r_t discount_t obs_t")
Params = collections.namedtuple("Params", "policy target_policy q target_q mpo")
TrainableParams = collections.namedtuple("TrainableParams", "policy q mpo")
MpoParams = collections.namedtuple("MpoParams", "temperature alpha")
ActorOutput = collections.namedtuple("ActorOutput", "actions logits")
LearnerState = collections.namedtuple("LearnerState", "count opt_state")

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 7, "Random seed.")
flags.DEFINE_integer("train_episodes", 400, "Number of train episodes.")
flags.DEFINE_integer("batch_size", 32, "Size of the training batch.")
flags.DEFINE_integer("replay_capacity", 2000, "Capacity of the replay buffer.")
flags.DEFINE_integer("hidden_units", 64, "Number of network hidden units.")
flags.DEFINE_integer(
    "num_action_samples", 4, "Number of actions to sample in the MPO E-step."
)
flags.DEFINE_integer("target_update_period", 50, "How often to update the target nets.")
flags.DEFINE_float("discount_factor", 0.99, "Return discount factor.")
flags.DEFINE_float("learning_rate", 3e-4, "Optimizer learning rate.")
flags.DEFINE_float("critic_loss_coef", 1.0, "Weight on the critic loss.")
flags.DEFINE_float("kl_epsilon", 0.1, "KL constraint epsilon for MPO.")
flags.DEFINE_float(
    "temperature_epsilon", 0.1, "Temperature constraint epsilon for MPO."
)
flags.DEFINE_float("init_temperature", 1.0, "Initial MPO temperature value.")
flags.DEFINE_float("init_alpha", 1.0, "Initial MPO KL dual variable.")
flags.DEFINE_integer("eval_episodes", 50, "Number of evaluation episodes.")
flags.DEFINE_integer("evaluate_every", 50, "Number of episodes between evaluations.")


def build_policy_network(num_actions: int) -> hk.Transformed:
    """Factory for the policy network producing action logits."""

    def forward(obs):
        network = hk.Sequential(
            [hk.Flatten(), nets.MLP([FLAGS.hidden_units, num_actions])]
        )
        return network(obs)

    return hk.without_apply_rng(hk.transform(forward))


def build_q_network(num_actions: int) -> hk.Transformed:
    """Factory for the critic network producing Q-values."""

    def forward(obs):
        network = hk.Sequential(
            [hk.Flatten(), nets.MLP([FLAGS.hidden_units, num_actions])]
        )
        return network(obs)

    return hk.without_apply_rng(hk.transform(forward))


class ReplayBuffer:
    """A simple Python replay buffer."""

    def __init__(self, capacity):
        self._prev = None
        self._action = None
        self._latest = None
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, env_output, action):
        self._prev = self._latest
        self._action = action
        self._latest = env_output

        if action is not None:
            self.buffer.append(
                (
                    self._prev.observation,
                    self._action,
                    self._latest.reward,
                    self._latest.discount,
                    self._latest.observation,
                )
            )

    def sample(self, batch_size):
        obs_tm1, a_tm1, r_t, discount_t, obs_t = zip(
            *random.sample(self.buffer, batch_size)
        )
        discount_t = np.asarray(discount_t) * FLAGS.discount_factor
        return Batch(
            obs_tm1=jnp.stack(obs_tm1),
            a_tm1=jnp.asarray(a_tm1),
            r_t=jnp.asarray(r_t),
            discount_t=jnp.asarray(discount_t),
            obs_t=jnp.stack(obs_t),
        )

    def is_ready(self, batch_size):
        return batch_size <= len(self.buffer)


class MPOAgent:
    """Maximum a posteriori policy optimization for discrete control."""

    def __init__(self, observation_spec, action_spec):
        self._observation_spec = observation_spec
        self._action_spec = action_spec
        self._num_action_samples = FLAGS.num_action_samples
        self._critic_coef = FLAGS.critic_loss_coef
        self._target_update_period = FLAGS.target_update_period
        self._init_temperature = FLAGS.init_temperature
        self._init_alpha = FLAGS.init_alpha
        self._temperature_epsilon = FLAGS.temperature_epsilon
        self._kl_epsilon = FLAGS.kl_epsilon

        self._policy_network = build_policy_network(action_spec.num_values)
        self._q_network = build_q_network(action_spec.num_values)
        self._optimizer = optax.adam(FLAGS.learning_rate)
        self._td_error = jax.vmap(
            functools.partial(rlax.q_learning, stop_target_gradients=True)
        )

        self.actor_step = jax.jit(self.actor_step)
        self.learner_step = jax.jit(self.learner_step)

    def initial_params(self, key):
        sample_input = self._observation_spec.generate_value()
        sample_input = jnp.expand_dims(sample_input, 0)
        key_seq = hk.PRNGSequence(key)
        policy_params = self._policy_network.init(next(key_seq), sample_input)
        q_params = self._q_network.init(next(key_seq), sample_input)
        mpo_params = MpoParams(
            temperature=jnp.array(self._init_temperature, dtype=jnp.float32),
            alpha=jnp.array(self._init_alpha, dtype=jnp.float32),
        )
        return Params(policy_params, policy_params, q_params, q_params, mpo_params)

    def initial_actor_state(self):
        return ()

    def initial_learner_state(self, params):
        trainable = TrainableParams(policy=params.policy, q=params.q, mpo=params.mpo)
        opt_state = self._optimizer.init(trainable)
        return LearnerState(count=jnp.zeros((), dtype=jnp.int32), opt_state=opt_state)

    def actor_step(self, params, env_output, actor_state, key, evaluation):
        obs = jnp.expand_dims(env_output.observation, 0)
        logits = self._policy_network.apply(params.policy, obs)[0]
        train_a = rlax.softmax().sample(key, logits)
        eval_a = rlax.greedy().sample(key, logits)
        action = jax.lax.select(evaluation, eval_a, train_a)
        return ActorOutput(actions=action, logits=logits), actor_state

    def learner_step(self, params, data, learner_state, key):
        trainable = TrainableParams(policy=params.policy, q=params.q, mpo=params.mpo)
        grad_fn = jax.grad(self._loss, has_aux=True)
        grads, _ = grad_fn(trainable, params.target_policy, params.target_q, data, key)
        updates, opt_state = self._optimizer.update(
            grads, learner_state.opt_state, trainable
        )
        trainable = optax.apply_updates(trainable, updates)
        count = learner_state.count + 1
        target_policy = optax.periodic_update(
            trainable.policy, params.target_policy, count, self._target_update_period
        )
        target_q = optax.periodic_update(
            trainable.q, params.target_q, count, self._target_update_period
        )
        params = Params(
            policy=trainable.policy,
            target_policy=target_policy,
            q=trainable.q,
            target_q=target_q,
            mpo=trainable.mpo,
        )
        return params, LearnerState(count=count, opt_state=opt_state)

    def _loss(self, trainable, target_policy, target_q, batch, key):
        policy_logits = self._policy_network.apply(trainable.policy, batch.obs_tm1)
        target_policy_logits = jax.lax.stop_gradient(
            self._policy_network.apply(target_policy, batch.obs_tm1)
        )
        q_tm1 = self._q_network.apply(trainable.q, batch.obs_tm1)
        q_t = self._q_network.apply(target_q, batch.obs_t)

        td_errors = self._td_error(q_tm1, batch.a_tm1, batch.r_t, batch.discount_t, q_t)
        critic_loss = jnp.mean(rlax.l2_loss(td_errors))

        sample_key, _ = jax.random.split(key)
        sample_keys = jax.random.split(sample_key, self._num_action_samples)
        sample_actions = jax.vmap(
            lambda k: jax.random.categorical(k, target_policy_logits, axis=-1)
        )(sample_keys)
        log_probs = jax.nn.log_softmax(policy_logits)
        sample_log_probs = jnp.take_along_axis(
            log_probs[None, ...], sample_actions[..., None], axis=-1
        )[..., 0]
        sample_q_values = jnp.take_along_axis(
            q_tm1[None, ...], sample_actions[..., None], axis=-1
        )[..., 0]

        kl = rlax.categorical_kl_divergence(policy_logits, target_policy_logits)
        kl_constraints = [
            (
                kl,
                rlax.LagrangePenalty(
                    alpha=trainable.mpo.alpha, epsilon=self._kl_epsilon
                ),
            )
        ]
        temperature_constraint = rlax.LagrangePenalty(
            alpha=trainable.mpo.temperature, epsilon=self._temperature_epsilon
        )
        mpo_loss, _ = rlax.mpo_loss(
            sample_log_probs=sample_log_probs,
            sample_q_values=sample_q_values,
            temperature_constraint=temperature_constraint,
            kl_constraints=kl_constraints,
            sample_axis=0,
        )
        total_loss = jnp.mean(mpo_loss) + self._critic_coef * critic_loss
        return total_loss, {"critic_loss": critic_loss}


def main(unused_arg):
    # env = catch.Catch(seed=FLAGS.seed)
    # env = cartpole.Cartpole(seed=FLAGS.seed)
    env = mountain_car.MountainCar(seed=FLAGS.seed)
    agent = MPOAgent(
        observation_spec=env.observation_spec(), action_spec=env.action_spec()
    )
    accumulator = ReplayBuffer(FLAGS.replay_capacity)
    experiment.run_loop(
        agent=agent,
        environment=env,
        accumulator=accumulator,
        seed=FLAGS.seed,
        batch_size=FLAGS.batch_size,
        train_episodes=FLAGS.train_episodes,
        evaluate_every=FLAGS.evaluate_every,
        eval_episodes=FLAGS.eval_episodes,
    )


if __name__ == "__main__":
    app.run(main)
