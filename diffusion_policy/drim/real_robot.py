"""Run a DRIM checkpoint on the dressing rig.

This is deliberately a small adapter around :class:`drim.infer.DrimRunner`.
It owns the two RealSense cameras (like ``direct_realsense_joystick_collector``)
and uses the same ``FrankaRobotClient`` state and velocity API as the collector.

The checkpoint predicts a *position offset*, not a velocity.  ``DrimRunner``
adds the measured EE position back and returns an absolute centroid/setpoint.
The collector recorded a centroid position made by integrating joystick
velocity.  This adapter maintains the same centroid state: a bounded position
servo moves the state toward the predicted centroid, integrates its base
velocity, then adds the separately known zigzag velocity around that integrated
centroid.  It never treats the network output itself as a velocity command.

Safety: observation-only is the default.  Hardware commands require
``--execute`` and an interactive confirmation.  Ctrl-C always sends ``stop``.
"""
from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from diffusion_policy.drim.infer import DrimRunner, Observation


class DirectRealSenseCamera:
    """Thread-safe latest RGB frame reader, matching the collector format."""

    def __init__(self, key: str, serial: str, width: int, height: int, fps: int):
        import pyrealsense2 as rs
        self.key, self.rs = key, rs
        self.pipe, self.config = rs.pipeline(), rs.config()
        self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self.lock, self.stop_event = threading.Lock(), threading.Event()
        self.image: Optional[np.ndarray] = None
        self.stamp: Optional[float] = None
        self.frame_number: Optional[int] = None
        self.error: Optional[str] = None
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.profile = self.pipe.start(self.config)
        print(f"[{self.key}] direct RealSense controls: {self.controls()}",
              flush=True)
        print(f"[{self.key}] RGB control ranges (min,max,step): "
              f"{self.control_ranges()}", flush=True)
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name=f"drim-{self.key}")
        self.thread.start()

    def controls(self) -> Dict[str, float | str]:
        """Report the color-sensor controls that affect RGB distribution."""
        try:
            sensor = self._color_sensor()
        except RuntimeError:
            return {"error": "no color sensor with exposure control"}
        options = {
            "auto_exposure": self.rs.option.enable_auto_exposure,
            "exposure": self.rs.option.exposure,
            "gain": self.rs.option.gain,
            "auto_white_balance": self.rs.option.enable_auto_white_balance,
            "white_balance": self.rs.option.white_balance,
        }
        out = {name: float(sensor.get_option(option))
               for name, option in options.items() if sensor.supports(option)}
        out["sensor"] = sensor.get_info(self.rs.camera_info.name)
        return out

    def _color_sensor(self):
        sensors = self.profile.get_device().query_sensors()
        # Both the depth and RGB sensors may support exposure/gain. White
        # balance is RGB-specific, so it is the reliable discriminator.
        for candidate in sensors:
            if candidate.supports(self.rs.option.white_balance):
                return candidate
        for candidate in sensors:
            name = candidate.get_info(self.rs.camera_info.name).lower()
            if ("rgb" in name or "color" in name) and candidate.supports(
                    self.rs.option.exposure):
                return candidate
        raise RuntimeError(f"{self.key} has no color sensor with exposure control")

    def control_ranges(self) -> Dict[str, tuple]:
        sensor = self._color_sensor()
        options = {
            "exposure": self.rs.option.exposure,
            "gain": self.rs.option.gain,
            "white_balance": self.rs.option.white_balance,
        }
        return {name: (float(r.min), float(r.max), float(r.step))
                for name, option in options.items() if sensor.supports(option)
                for r in (sensor.get_option_range(option),)}

    def set_fixed_controls(self, exposure: float, gain: float,
                           white_balance: float) -> None:
        """Disable RGB auto-controls and apply a reproducible color profile."""
        sensor = self._color_sensor()
        for option, value in (
                (self.rs.option.enable_auto_exposure, 0.0),
                (self.rs.option.exposure, exposure),
                (self.rs.option.gain, gain),
                (self.rs.option.enable_auto_white_balance, 0.0),
                (self.rs.option.white_balance, white_balance)):
            if sensor.supports(option):
                bounds = sensor.get_option_range(option)
                if not bounds.min <= value <= bounds.max:
                    raise ValueError(
                        f"{self.key} {option}={value} is outside the RGB sensor "
                        f"range [{bounds.min}, {bounds.max}] (step {bounds.step})")
                sensor.set_option(option, float(value))

    def lock_current_controls(self) -> None:
        """Freeze the settled auto values so they cannot drift during a trial."""
        current = self.controls()
        self.set_fixed_controls(current.get("exposure", 120.0),
                                current.get("gain", 0.0),
                                current.get("white_balance", 5900.0))

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                fs = self.pipe.wait_for_frames(timeout_ms=1000)
                f = fs.get_color_frame()
                if f:
                    with self.lock:
                        self.image = np.asanyarray(f.get_data()).copy()
                        self.stamp = f.get_timestamp() / 1000.0
                        self.frame_number = int(f.get_frame_number())
            except RuntimeError as exc:
                if not self.stop_event.is_set():
                    self.error = str(exc)
                    self.stop_event.set()

    def latest(self) -> np.ndarray:
        return self.latest_with_stamp()[0]

    def latest_with_stamp(self) -> tuple[np.ndarray, float, int]:
        """Return an atomic RGB frame/timestamp/RealSense-number triple."""
        with self.lock:
            if self.image is None or self.stamp is None or self.frame_number is None:
                raise RuntimeError(f"no frame yet from {self.key}")
            return self.image.copy(), float(self.stamp), int(self.frame_number)

    def wait_for_frame(self, timeout: float = 15.0) -> None:
        """Wait for the first directly captured RGB frame before inference."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if self.image is not None:
                    return
            if self.stop_event.is_set():
                raise RuntimeError(f"direct RealSense {self.key} stopped: "
                                   f"{self.error or 'unknown capture error'}")
            time.sleep(.05)
        raise TimeoutError(
            f"timed out waiting for direct RealSense frame from {self.key}; "
            "check its serial, USB connection, and that no collector or other "
            "pyrealsense process owns the device")

    def close(self) -> None:
        self.stop_event.set()
        try:
            self.pipe.stop()
        except RuntimeError:
            pass
        if self.thread:
            self.thread.join(timeout=2.0)


@dataclass
class Zigzag:
    """The collector's X-axis primitive, expressed in base-frame m/s."""

    amplitude: float = 0.05
    speed: float = 0.07
    direction: float = 0.0

    def reset(self) -> None:
        self.direction = 0.0

    def command(self, current_x: float, centroid_x: float) -> np.ndarray:
        half = self.amplitude / 2.0
        if self.amplitude <= 0:
            self.direction = 0.0
        elif current_x - centroid_x >= half:
            self.direction = -self.speed
        elif current_x - centroid_x <= -half or self.direction == 0:
            self.direction = self.speed
        return np.array([self.direction, 0., 0., 0., 0., 0.], np.float32)


def setpoint_velocity(target: np.ndarray, measured: np.ndarray, gain: float,
                      limits: Sequence[float]) -> np.ndarray:
    """Bounded Cartesian P tracking for an absolute DRIM position target."""
    target, measured = np.asarray(target, np.float32), np.asarray(measured, np.float32)
    lim = np.asarray(limits, np.float32)
    if target.shape != (3,) or measured.shape != (3,) or lim.shape != (3,):
        raise ValueError("target, measured, and limits must each be 3-vectors")
    return np.clip(float(gain) * (target - measured), -lim, lim)


def centroid_velocity(target: np.ndarray, previous_target: Optional[np.ndarray],
                      dt: float, limits: Sequence[float]) -> np.ndarray:
    """Legacy diagnostic: finite-difference successive predicted centroids.

    This is appropriate for *recorded, low-noise* centroids but not for a live
    policy's independently predicted centroids: its millimetre prediction
    error becomes a large velocity after division by ``dt``.  Live execution
    uses the centroid-state servo below by default.
    """
    target = np.asarray(target, np.float32)
    lim = np.asarray(limits, np.float32)
    if target.shape != (3,) or lim.shape != (3,) or dt <= 0:
        raise ValueError("target/limits must be 3-vectors and dt must be positive")
    if previous_target is None:
        return np.zeros(3, np.float32)
    return np.clip((target - np.asarray(previous_target, np.float32)) / float(dt),
                   -lim, lim)


def centroid_servo_velocity(target: np.ndarray, centroid: np.ndarray, gain: float,
                            limits: Sequence[float]) -> np.ndarray:
    """Bounded base velocity toward a predicted, zigzag-free centroid.

    ``centroid`` is a controller state, initialized from the measured EE pose
    once and subsequently updated with ``centroid += velocity * dt``.  This is
    the inverse *state structure* of collection, rather than a derivative of
    noisy, independently predicted position samples.
    """
    return setpoint_velocity(target, centroid, gain, limits)


def executed_command(policy_velocity: np.ndarray,
                     zigzag_velocity: np.ndarray) -> np.ndarray:
    """Return the 6-D Cartesian velocity that reaches the robot server.

    The collector's unfortunately named ``zigzag_action`` field records this
    *complete* command, not the X zigzag term alone: it writes
    ``command_linear`` after the primitive was added.  The dynamics model was
    trained on that field, so its inference history must receive exactly the
    command sent on the preceding transition.
    """
    policy_velocity = np.asarray(policy_velocity, np.float32)
    zigzag_velocity = np.asarray(zigzag_velocity, np.float32)
    if policy_velocity.shape != (3,) or zigzag_velocity.shape != (6,):
        raise ValueError("policy velocity must be (3,), zigzag velocity must be (6,)")
    return np.concatenate([policy_velocity + zigzag_velocity[:3],
                           np.zeros(3, np.float32)])


def save_roi_debug(cameras: Dict[str, DirectRealSenseCamera], runner,
                   output_dir: str) -> None:
    """Save raw, annotated, and exact policy-input RGB frames for inspection."""
    import cv2
    from diffusion_policy.drim.infer import crop_resize

    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    size = (runner.spec.image_shape[1], runner.spec.image_shape[2])
    for name, camera in cameras.items():
        rgb = camera.latest()
        overlay = rgb.copy()
        box = runner.roi.get(name)
        if box is not None:
            y, x, h, w = (int(v) for v in box)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 3)
        crop = crop_resize(rgb, box, size)
        # OpenCV writes BGR.  ``*_rgb_interpreted`` is correct when the bytes
        # are RGB (the requested/collector format); ``*_bgr_interpreted`` is a
        # diagnostic only, for comparison with RealSense Viewer.
        cv2.imwrite(str(out / f"{name}_full.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"{name}_roi_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"{name}_policy_input.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"{name}_rgb_interpreted.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"{name}_bgr_interpreted.png"), rgb)
    print(f"Saved direct-camera ROI debug images to {out}", flush=True)


def _robot_observation(robot, cameras: Dict[str, DirectRealSenseCamera],
                       zigzag_action: np.ndarray, dt: float,
                       images: Optional[Dict[str, np.ndarray]] = None):
    # Kept local so the pure command helpers above can be smoke-tested on a
    # machine without the model's optional training dependencies.
    from diffusion_policy.drim.infer import Observation
    q = robot.get_joint_positions()
    pose = robot.get_ee_pose()
    twist = robot.get_ee_twist()
    wrench = robot.get_ee_wrench_base()
    diag = robot.get_teleop_diagnostics()
    if q is None or pose is None or twist is None or wrench is None or diag is None:
        raise RuntimeError("incomplete robot state (q/pose/twist/wrench/diagnostics required)")
    pos, quat = pose
    lin, ang = twist
    force, torque = wrench
    return Observation(
        q=np.asarray(q, np.float32), dq=np.asarray(diag["joint_velocity"], np.float32),
        ee_pos=np.asarray(pos, np.float32), ee_quat=np.asarray(quat, np.float32),
        ee_twist=np.concatenate([lin, ang]).astype(np.float32),
        wrench=np.concatenate([force, torque]).astype(np.float32),
        images=(images if images is not None
                else {k: c.latest() for k, c in cameras.items()}),
        zigzag_action=np.asarray(zigzag_action, np.float32), dt=float(dt))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="DRIM run directory")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--stage", default="D2", choices=("B0", "B1", "D2"))
    p.add_argument("--flow-steps", type=int, default=None,
                   help="experimental Euler sampling steps for the loaded flow "
                        "policy; defaults to the checkpoint's value (10)")
    p.add_argument("--front-serial", default="117122250054")
    p.add_argument("--back-serial", default="036522071464")
    p.add_argument("--camera-fps", type=int, default=15)
    p.add_argument("--fail-on-camera-drop", action="store_true",
                   help="stop before inference if either direct RealSense stream "
                        "skips a frame number; use to validate the policy input stream")
    p.add_argument("--debug-roi-dir", default=None,
                   help="save direct full frames, ROI overlays, and exact policy inputs here")
    p.add_argument("--camera-channel-order", choices=("rgb", "bgr"), default="rgb",
                   help="byte order presented to the policy; rgb matches the collector")
    p.add_argument("--camera-profile", choices=("auto", "lock", "fixed"),
                   default="fixed",
                   help="auto leaves controls dynamic; lock warms then freezes them; "
                        "fixed applies exposure/gain/white-balance values")
    p.add_argument("--camera-warmup-seconds", type=float, default=3.0,
                   help="auto-control settling time before a lock profile")
    p.add_argument("--camera-exposure", type=float, default=100.0,
                   help="fixed profile exposure (RealSense units)")
    p.add_argument("--camera-gain", type=float, default=0.0,
                   help="fixed profile analog gain")
    p.add_argument("--camera-white-balance", type=float, default=3500.0,
                   help="fixed profile white balance (Kelvin)")
    p.add_argument("--control-hz", type=float, default=15.0)
    p.add_argument("--warmup-inference-steps", type=int, default=1,
                   help="no-motion GPU warm-up steps before arming, followed by "
                        "a runner reset (default: 1)")
    p.add_argument("--seconds", type=float, default=0., help="0 means run until Ctrl-C")
    p.add_argument("--position-gain", "--centroid-gain", dest="position_gain",
                   type=float, default=1.0,
                   help="centroid-servo gain in 1/s (default: 1.0)")
    p.add_argument("--execution-mode",
                   choices=("centroid_servo", "centroid_velocity", "position_p"),
                   default="centroid_servo",
                   help="centroid_servo maintains/integrates a centroid state "
                        "(default); centroid_velocity is legacy finite "
                        "differencing for diagnosis only; position_p tracks EE "
                        "directly and is an experiment")
    p.add_argument("--max-policy-speed", type=float, nargs=3,
                   # The collector's joystick component is bounded by
                   # LIN_SCALE=0.01 m/s; adding its 0.07 m/s zigzag gives the
                   # demonstrated 0.08 m/s X envelope.  Do not silently turn
                   # a setpoint tracker into a faster controller at deploy.
                   default=(0.01, 0.01, 0.01), metavar=("VX", "VY", "VZ"))
    p.add_argument("--zigzag-amplitude", type=float, default=.05)
    p.add_argument("--zigzag-speed", type=float, default=.07)
    p.add_argument("--no-zigzag", action="store_true")
    p.add_argument("--zero-policy-output", action="store_true",
                   help="hold the initial centroid and execute only zigzag; "
                        "still prints the unmasked policy target")
    p.add_argument("--policy-axis-mask", type=float, nargs=3,
                   default=(1., 1., 1.), metavar=("MX", "MY", "MZ"),
                   help="centroid-policy authority per XYZ axis; 0 holds its "
                        "initial centroid and removes its velocity contribution")
    p.add_argument("--execute", action="store_true",
                   help="send velocity commands (otherwise observe/infer only)")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    if (a.control_hz <= 0 or a.camera_fps <= 0 or a.position_gain <= 0
            or a.camera_warmup_seconds < 0):
        raise ValueError("control/camera rates and position gain must be positive")
    if a.warmup_inference_steps < 0:
        raise ValueError("--warmup-inference-steps must be non-negative")
    if a.flow_steps is not None and a.flow_steps <= 0:
        raise ValueError("--flow-steps must be positive")
    if any(v < 0 or v > 1 for v in a.policy_axis_mask):
        raise ValueError("--policy-axis-mask values must lie in [0, 1]")
    # Defer ROS imports: --help and import-time unit tests do not need hardware.
    import rospy
    from bimanual_dressing.control.client import FrankaRobotClient
    from diffusion_policy.drim.infer import DrimRunner

    rospy.init_node("drim_real_robot", anonymous=True)
    runner = DrimRunner.load(a.run, device=a.device, upto=a.stage)
    if a.flow_steps is not None:
        original_flow_steps = runner.model.slow.n_flow_steps
        runner.model.slow.n_flow_steps = int(a.flow_steps)
        print(f"Experimental sampler override: {original_flow_steps} -> "
              f"{runner.model.slow.n_flow_steps} flow steps", flush=True)
    sp = runner.spec
    if sp.action_mode != "delta_ee_pos":
        raise RuntimeError(f"this adapter is for delta_ee_pos checkpoints, got {sp.action_mode}")
    if tuple(sp.cameras) != ("image_bed_front", "image_bed_back"):
        raise RuntimeError(f"unexpected camera contract {sp.cameras}")
    if sp.act_dim != 3 or sp.exo_dim != 6:
        raise RuntimeError(f"expected 3-D target and 6-D zigzag, got {sp.act_dim}/{sp.exo_dim}")

    robot = FrankaRobotClient(robot_name="arm1")
    cameras = {
        "image_bed_front": DirectRealSenseCamera("image_bed_front", a.front_serial, 640, 480, a.camera_fps),
        "image_bed_back": DirectRealSenseCamera("image_bed_back", a.back_serial, 640, 480, a.camera_fps),
    }
    for cam in cameras.values():
        cam.start()
    try:
        # ``pipeline.start`` means the device was claimed, not that a frame is
        # ready.  The collector avoids this race by simply waiting in its event
        # loop; inference must make that barrier explicit before the first
        # policy call.
        for cam in cameras.values():
            print(f"Waiting for direct RealSense frame: {cam.key}", flush=True)
            cam.wait_for_frame()
        if a.camera_profile == "fixed":
            for cam in cameras.values():
                cam.set_fixed_controls(a.camera_exposure, a.camera_gain,
                                       a.camera_white_balance)
            # Let the sensor publish frames under the newly applied controls.
            time.sleep(.5)
        elif a.camera_profile == "lock":
            print(f"Warming RealSense auto controls for {a.camera_warmup_seconds:.1f}s",
                  flush=True)
            time.sleep(a.camera_warmup_seconds)
            for cam in cameras.values():
                cam.lock_current_controls()
        for cam in cameras.values():
            print(f"[{cam.key}] active camera controls: {cam.controls()}", flush=True)
        if a.debug_roi_dir:
            save_roi_debug(cameras, runner, a.debug_roi_dir)
    except Exception:
        for cam in cameras.values():
            cam.close()
        raise
    if a.warmup_inference_steps:
        # The first CUDA sampling call lazily initializes kernels and can take
        # several camera periods.  Do it before the episode and discard its
        # history, otherwise the strict stream check reports a startup drop
        # that cannot affect a real action anyway.
        print(f"Warming policy for {a.warmup_inference_steps} no-motion step(s)",
              flush=True)
        warm_command = np.zeros(6, np.float32)
        for _ in range(a.warmup_inference_steps):
            warm_start = time.monotonic()
            warm_images = {k: c.latest() for k, c in cameras.items()}
            warm_obs = _robot_observation(robot, cameras, warm_command,
                                           1.0 / a.control_hz, warm_images)
            runner.step(warm_obs)
            print(f"Policy warm-up took {(time.monotonic() - warm_start) * 1000:.0f}ms",
                  flush=True)
        runner.reset()
    if a.execute:
        input("DRIM will command arm1. Verify the emergency stop, then press Enter to arm. ")
    else:
        print("Dry run: inferring targets only; no robot command will be sent.")

    primitive = Zigzag(0. if a.no_zigzag else a.zigzag_amplitude, a.zigzag_speed)
    runner.reset(); primitive.reset()
    dt_nominal, previous, start = 1.0 / a.control_hz, time.monotonic(), time.monotonic()
    # This is the prior *complete* velocity command.  It is passed into the
    # runner at t because its transition is (state[t-1], command[t-1], state[t]).
    previous_command = np.zeros(6, np.float32)
    axis_mask = np.asarray((0., 0., 0.) if a.zero_policy_output
                           else a.policy_axis_mask, np.float32)
    held_centroid: Optional[np.ndarray] = None
    # This is the collector's zigzag-free centroid state.  It must not be
    # replaced with the measured EE pose: the latter is intentionally offset
    # and oscillating in X due to the primitive.
    controlled_centroid: Optional[np.ndarray] = None
    previous_target: Optional[np.ndarray] = None
    previous_camera_stamps: Optional[Dict[str, float]] = None
    previous_camera_numbers: Optional[Dict[str, int]] = None
    print("Live units: positions=m; model/servo errors=mm; velocities=mm/s. "
          f"mode={a.execution_mode} gain={a.position_gain:.2f}/s "
          f"limits={np.asarray(a.max_policy_speed) * 1000}mm/s "
          f"policy_mask={axis_mask}", flush=True)
    try:
        while not rospy.is_shutdown():
            if robot.get_last_error():
                raise RuntimeError("robot server reported an error")
            now = time.monotonic()
            if a.seconds > 0 and now - start >= a.seconds:
                break
            # Collection appended a sample only once *both* direct-camera
            # streams had advanced.  Reusing an image here would turn the
            # checkpoint's [-1, 0] visual context into [t, t] or [t-1, t-1].
            # Snapshot first, then gate the whole observation on a fresh pair.
            camera_pairs = {k: c.latest_with_stamp() for k, c in cameras.items()}
            camera_stamps = {k: pair[1] for k, pair in camera_pairs.items()}
            camera_numbers = {k: pair[2] for k, pair in camera_pairs.items()}
            if (previous_camera_stamps is not None and any(
                    camera_stamps[k] <= previous_camera_stamps[k]
                    for k in camera_stamps)):
                time.sleep(.002)
                continue
            frame_gaps = ({k: 0 for k in camera_numbers}
                          if previous_camera_numbers is None else
                          {k: max(0, camera_numbers[k] - previous_camera_numbers[k] - 1)
                           for k in camera_numbers})
            if a.fail_on_camera_drop and any(frame_gaps.values()):
                raise RuntimeError(
                    "camera frame drop before policy input: "
                    + ", ".join(f"{k} skipped {n}" for k, n in frame_gaps.items()
                                if n))
            images = {k: pair[0] for k, pair in camera_pairs.items()}
            # Advance the control clock only for an accepted camera pair.  A
            # polling retry must not turn a 15-Hz integration interval into a
            # 2-ms one.
            dt = min(max(now - previous, 1e-3), .5); previous = now
            obs = _robot_observation(robot, cameras, previous_command, dt, images)
            if a.camera_channel_order == "bgr":
                obs.images = {k: v[..., ::-1].copy() for k, v in obs.images.items()}
            # A masked axis holds a *fixed* initial centroid.  Replacing it
            # with the current pose every iteration would make that centroid
            # follow the zigzag and prevent its position-based reversals.
            if held_centroid is None:
                held_centroid = obs.ee_pos.copy()
            if controlled_centroid is None:
                controlled_centroid = obs.ee_pos.copy()
            inference_start = time.monotonic()
            target = runner.step(
                obs, target_transform=lambda raw: held_centroid + axis_mask *
                (raw - held_centroid))
            inference_ms = (time.monotonic() - inference_start) * 1000.0
            if a.execution_mode == "centroid_servo":
                policy_vel = centroid_servo_velocity(target, controlled_centroid,
                                                     a.position_gain,
                                                     a.max_policy_speed)
            elif a.execution_mode == "centroid_velocity":
                centroid_step = (np.zeros(3, np.float32) if previous_target is None
                                 else target - previous_target)
                policy_vel = centroid_velocity(target, previous_target, obs.dt,
                                               a.max_policy_speed)
            else:
                centroid_step = target - obs.ee_pos
                policy_vel = setpoint_velocity(target, obs.ee_pos, a.position_gain,
                                               a.max_policy_speed)
            # A zero mask is *policy velocity exactly zero*, not merely a zero
            # target fed through a P controller.  Apply it before updating the
            # controller state so a masked centroid remains fixed.
            policy_vel *= axis_mask
            if a.execution_mode == "centroid_servo":
                centroid_step = policy_vel * obs.dt
            controlled_centroid += policy_vel * obs.dt
            # The primitive's reference is the maintained centroid, not the
            # noisy instantaneous policy sample.  This is precisely the
            # collector's ``current_x - centroid_x`` feedback quantity.
            exo = primitive.command(obs.ee_pos[0], controlled_centroid[0])
            command = executed_command(policy_vel, exo)
            if a.execution_mode == "centroid_servo":
                # ``hist_a`` was trained on the collector's integrated
                # centroid, not on an unexecuted/noisy policy proposal.  The
                # runner has already staged this step, so replace that one
                # pending action before the next observation closes it.
                runner.replace_pending_action(controlled_centroid)
            # Likewise, training's ``hist_exo[t]`` is the complete command
            # actually sent at t.  It is only available after adding the
            # primitive, so replace the staged placeholder before t+1 closes
            # this transition.
            runner.replace_pending_exo(command)
            if a.execute:
                robot.set_ee_velocity(command[:3].tolist(), command[3:].tolist())
            status = "R" if runner.last_replanned else "-"
            if runner.stage == "D2":
                message = "D2" if runner.message_ready else "B1-warmup"
            else:
                message = runner.stage
            def vec(v, scale=1.0, precision=1):
                return np.array2string(np.asarray(v) * scale, precision=precision,
                                       suppress_small=True, separator=",")
            print(f"step={runner._k - 1:04d} {status} "
                  f"chunk={runner.last_chunk_index}/{runner.spec.exec_horizon - 1} "
                  f"{message} infer={inference_ms:.0f}ms "
                  f"cam=F{camera_numbers['image_bed_front']}+{frame_gaps['image_bed_front']}/"
                  f"B{camera_numbers['image_bed_back']}+{frame_gaps['image_bed_back']}",
                  flush=True)
            print(f"  ee_pose [m]:       {vec(obs.ee_pos, precision=4)}", flush=True)
            # The checkpoint is trained on delta_action = centroid - EE.  Show
            # that raw network quantity rather than the reconstructed centroid
            # so its sign cannot be mistaken for a velocity direction.
            print(f"  delta_action [m]:  "
                  f"{vec(runner.last_policy_target - obs.ee_pos, precision=4)}",
                  flush=True)
            print(f"  sent_v [mm/s]:     {vec(command[:3], 1000)} "
                  f"(policy={vec(policy_vel, 1000)}, zigzag_x={exo[0] * 1000:+.1f})",
                  flush=True)
            previous_command = command
            previous_target = target.copy()
            previous_camera_stamps = camera_stamps
            previous_camera_numbers = camera_numbers
            time.sleep(max(0., dt_nominal - (time.monotonic() - now)))
    except KeyboardInterrupt:
        pass
    finally:
        if a.execute:
            robot.stop()
        for cam in cameras.values():
            cam.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
