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


def _spec(cameras, prop_fields):
    from diffusion_policy.drim.spec import DrimSpec
    return DrimSpec(prop_dim=sum({"ee_pos": 3, "ee_lin_vel": 3, "q": 7, "dq": 7,
                                  "ee_quat": 4, "ee_twist": 6}[f]
                                 for f in prop_fields),
                    act_dim=3, wrench_dim=6, cameras=tuple(cameras),
                    image_shape=(3, 160, 128), crop_shape=(144, 115),
                    pred_horizon=16, exec_horizon=8, message_window=32,
                    message_dim=16, fast_limits=(.25,)*3,
                    action_mode="delta_action", exo_dim=6,
                    prop_fields=tuple(prop_fields), wrench_fields=("wrench",))


def test_the_adapter_opens_the_cameras_the_checkpoint_names():
    """A one-camera run is an ablation, not a malformed checkpoint.

    Hardcoding the pair here would mean that if the bed-back view turns out not
    to earn its place, the answer could never be deployed.
    """
    import inspect
    from diffusion_policy.drim import real_robot as RR
    src = inspect.getsource(RR.main)
    assert "for c in sp.cameras" in src, (
        "the adapter must build its cameras from the checkpoint's contract")
    assert '!= ("image_bed_front", "image_bed_back")' not in src, (
        "the fixed two-camera check is back; a 1cam checkpoint cannot start")
    #: a camera the adapter has no serial for is refused, not skipped
    assert "no serial for" in src


def test_the_robot_observation_serves_both_state_vectors():
    """The rig publishes one 6-vector twist; the current default wants its
    linear half under a different name.

    ``ee_lin_vel`` is never published — the loader carves it out of ``ee_twist``
    when the angular half is excluded, which it is by default, being identically
    zero on all 94,611 recorded steps. Without this the adapter could build an
    Observation that no current checkpoint can read.
    """
    from diffusion_policy.drim.infer import Observation
    obs = Observation(
        q=np.arange(7, dtype=np.float32), dq=np.arange(7, dtype=np.float32) + 10,
        ee_pos=np.float32([.5, .1, .6]), ee_quat=np.float32([0, 0, 0, 1]),
        ee_twist=np.arange(6, dtype=np.float32) + 100,
        wrench=np.arange(6, dtype=np.float32) + 200,
        images={}, zigzag_action=np.zeros(6, np.float32))

    #: the current default: position and its velocity, no joint space
    now = obs.prop(_spec(("image_bed_front",), ("ee_pos", "ee_lin_vel")))
    np.testing.assert_allclose(now, [.5, .1, .6, 100, 101, 102])

    #: and the --keep q,dq ablation, from the same Observation
    wide = obs.prop(_spec(("image_bed_front", "image_bed_back"),
                          ("q", "dq", "ee_pos", "ee_quat", "ee_twist")))
    assert len(wide) == 27 and wide[0] == 0 and wide[7] == 10

    #: a field the rig cannot supply stops the loop and names it, rather than
    #: arriving at the encoder as a silently shorter vector
    from diffusion_policy.drim.spec import DrimSpec
    import pytest
    cannot = DrimSpec(prop_dim=1, act_dim=3, wrench_dim=6,
                      cameras=("image_bed_front",), image_shape=(3, 160, 128),
                      crop_shape=(144, 115), pred_horizon=16, exec_horizon=8,
                      message_window=32, message_dim=16, fast_limits=(.25,)*3,
                      action_mode="delta_action", exo_dim=6,
                      prop_fields=("gripper_pos",), wrench_fields=("wrench",))
    with pytest.raises(AssertionError, match="gripper_pos"):
        obs.prop(cannot)
