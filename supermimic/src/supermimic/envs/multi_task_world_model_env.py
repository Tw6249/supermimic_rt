"""Multi-task differentiable world model env for concurrent encoder training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDict

from supermimic.envs.world_model_env import WorldModelEnv
from supermimic.tasks.task_spec import build_actor_obs_td, extract_goal

if TYPE_CHECKING:
  from supermimic.models import Dynamics
  from supermimic.replay_buffer import ReplayBuffer
  from supermimic.representations import StateRepresentation
  from supermimic.tasks.rt_4cam import StreamState
  from supermimic.tasks.task_spec import TaskSpec


class MultiTaskWorldModelEnv(WorldModelEnv):
  """WM env for motion tracking with per-slot goal modalities.

  Reward is motion-tracking for every slot; tasks differ only in what
  ``goal_<name>`` key the actor's encoder consumes. ``get_observations``
  emits all per-task goal keys full-batch and the actor slices its own
  slot range. The obs-history ring (history_length > 1) lives in the base.

  Stateful tasks (``TaskSpec.stateful``) keep a per-env goal state here: drawn
  fresh at ``reset``, advanced once per imagined step, and read together with
  the robot's measured joint positions when the observation is assembled.
  """

  def __init__(
    self,
    dynamics: Dynamics,
    representation: StateRepresentation,
    env: Any,
    batch_size: int,
    replay_buffer: ReplayBuffer,
    task_specs: list[TaskSpec],
    history_length: int,
    termination_softness: float,
    device: str = "cpu",
  ) -> None:
    self.task_specs = task_specs
    assert task_specs[0].name == "motion", "task_specs[0] must be the motion task"
    self._stateful_specs = [ts for ts in task_specs if ts.stateful]
    self._goal_states: dict[str, StreamState] = {}
    super().__init__(
      dynamics=dynamics,
      representation=representation,
      env=env,
      batch_size=batch_size,
      replay_buffer=replay_buffer,
      history_length=history_length,
      termination_softness=termination_softness,
      device=device,
    )

  def _build_task_goals(self) -> None:
    # The anchor task's goal is the actor goal already built in _update_goal; reuse it.
    anchor, *rest = self.task_specs
    self._task_goal_obs: dict[str, torch.Tensor] = {anchor.name: self._actor_goal_obs}
    for ts in rest:
      if ts.stateful:
        continue  # needs the post-step sensor; built in get_observations
      self._task_goal_obs[ts.name] = ts.extract_training_goal(
        self.motion, self.motion_times, self.aux_state
      ).clone()

  def _on_reset(self, batch: TensorDict) -> None:
    for ts in self._stateful_specs:
      assert ts.init_goal_state is not None
      self._goal_states[ts.name] = ts.init_goal_state(self.motion, batch["motion_time"])
    super()._on_reset(batch)

  def _advance_task_state(self) -> None:
    for ts in self._stateful_specs:
      assert ts.advance_goal_state is not None
      self._goal_states[ts.name] = ts.advance_goal_state(
        self._goal_states[ts.name], self.motion, self.motion_times
      )

  def get_observations(self) -> TensorDict:
    goals = dict(self._task_goal_obs)
    if self._stateful_specs:
      joint_pos = self.representation.measured_joint_pos(
        self._latest_sensor, self.joint_state
      )
      for ts in self._stateful_specs:
        goals[ts.name] = extract_goal(
          ts,
          self.motion,
          self.motion_times,
          self.aux_state,
          joint_pos,
          self._goal_states[ts.name],
        )
    return build_actor_obs_td(goals, self._sensor_history(), critic=self._critic_obs())
