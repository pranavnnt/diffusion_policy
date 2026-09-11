"""Pure (no ROS/camera/robot) invariants for the DRIM live adapter."""

import numpy as np
from types import SimpleNamespace

from diffusion_policy.drim.infer import DrimRunner
from diffusion_policy.drim.real_robot import (
    Zigzag,
    centroid_servo_velocity,
    executed_command,
)


def test_centroid_servo_integrates_a_bounded_base_velocity():
    """Do not turn independent centroid predictions into finite differences."""
    centroid = np.array([0.400, 0.200, 0.600], dtype=np.float32)
    prediction = np.array([0.410, 0.180, 0.600], dtype=np.float32)
    velocity = centroid_servo_velocity(prediction, centroid, gain=1.0,
                                       limits=(.01, .01, .01))
    np.testing.assert_allclose(velocity, [.01, -.01, 0.], atol=1e-7)

    centroid += velocity / 15.0
    # The controller state advances by the sent base velocity, as the
    # collector's centroid did; it does not jump to a noisy prediction.
    np.testing.assert_allclose(centroid, [.40066667, .19933333, .6], atol=1e-7)


def test_zigzag_is_added_after_centroid_base_velocity():
    centroid = np.array([.400, .2, .6], dtype=np.float32)
    primitive = Zigzag(amplitude=.05, speed=.07)
    zigzag = primitive.command(current_x=.400, centroid_x=centroid[0])
    command = executed_command(np.array([.01, -.005, .002], dtype=np.float32),
                               zigzag)
    np.testing.assert_allclose(command, [.08, -.005, .002, 0., 0., 0.])


def test_centroid_servo_replaces_the_pending_history_action():
    """D2 must see the executed centroid, as it did during collection."""
    runner = object.__new__(DrimRunner)
    runner.spec = SimpleNamespace(act_dim=3)
    y = np.zeros(2, dtype=np.float32)
    exo = np.zeros(6, dtype=np.float32)
    runner._prev = (y, np.array([1., 2., 3.], dtype=np.float32), exo, .07)
    executed_centroid = np.array([.4, .2, .6], dtype=np.float32)
    runner.replace_pending_action(executed_centroid)
    np.testing.assert_allclose(runner._prev[1], executed_centroid)
    np.testing.assert_allclose(runner._prev[0], y)
    np.testing.assert_allclose(runner._prev[2], exo)


def test_live_command_replaces_the_pending_history_exo():
    """The D2 transition at t must contain the command sent at t, not t-1."""
    runner = object.__new__(DrimRunner)
    runner.spec = SimpleNamespace(act_dim=3, exo_dim=6)
    y = np.zeros(2, dtype=np.float32)
    action = np.zeros(3, dtype=np.float32)
    runner._prev = (y, action, np.zeros(6, dtype=np.float32), .07)
    sent = np.array([.07, -.005, .003, 0., 0., 0.], dtype=np.float32)
    runner.replace_pending_exo(sent)
    np.testing.assert_allclose(runner._prev[2], sent)
    np.testing.assert_allclose(runner._prev[0], y)
    np.testing.assert_allclose(runner._prev[1], action)
