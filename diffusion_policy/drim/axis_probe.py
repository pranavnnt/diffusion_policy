"""Guarded keyboard probe for identifying arm1's physical +Y/-Y directions.

This does not load a policy, camera, or zigzag primitive.  It sends the same
``FrankaRobotClient.set_ee_velocity([vx, vy, vz])`` command used by both the
teleoperation collector and ``drim.real_robot``.  It is intentionally limited
to base-frame Y translation, so it can answer the coordinate-frame question
without a learned policy or an X zigzag obscuring the motion.

Run only with the usual robot infrastructure already up::

    python -m diffusion_policy.drim.axis_probe --execute

Hold ``w`` for +Y and ``s`` for -Y.  Releasing the key (i.e. stopping terminal
key-repeat) triggers the short command watchdog and stops the robot.  Space
stops immediately; ``q`` stops and exits.  The default speed is 2 mm/s, so a
one-second hold moves about 2 mm.  The script never commands X, Z, or rotation.
"""
from __future__ import annotations

import argparse
import curses
import time
from typing import Optional, Sequence

import numpy as np


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=0.002,
                        help="absolute Y speed in m/s (default: 0.002 = 2 mm/s)")
    parser.add_argument("--max-speed", type=float, default=0.005,
                        help="hard safety ceiling for --speed in m/s (default: 0.005)")
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--key-timeout", type=float, default=0.25,
                        help="seconds after the last w/s repeat before stop")
    parser.add_argument("--execute", action="store_true",
                        help="actually send commands; otherwise display only")
    return parser.parse_args(argv)


def _velocity(direction: int, speed: float) -> np.ndarray:
    """Return the sole permitted command: base-frame +/-Y translation."""
    return np.array([0.0, float(direction) * speed, 0.0], dtype=np.float64)


def _pose_text(robot) -> str:
    pose = robot.get_ee_pose()
    if pose is None:
        return "ee_pos unavailable"
    pos, _ = pose
    return "ee_pos=[% .4f % .4f % .4f]" % tuple(np.asarray(pos, float))


def _ui(stdscr, robot, args: argparse.Namespace) -> int:
    stdscr.nodelay(True)
    stdscr.keypad(True)
    curses.curs_set(0)
    direction = 0
    last_motion_key = -float("inf")
    was_moving = False
    period = 1.0 / args.control_hz
    try:
        while True:
            loop_start = time.monotonic()
            key = stdscr.getch()
            if key in (ord("q"), ord("Q")):
                return 0
            if key in (ord("w"), ord("W")):
                direction, last_motion_key = +1, loop_start
            elif key in (ord("s"), ord("S")):
                direction, last_motion_key = -1, loop_start
            elif key == ord(" "):
                direction = 0
                last_motion_key = -float("inf")

            active = direction if loop_start - last_motion_key <= args.key_timeout else 0
            velocity = _velocity(active, args.speed)
            if args.execute and active:
                if robot.get_last_error():
                    raise RuntimeError("robot server reported an error")
                robot.set_ee_velocity(velocity.tolist(), [0.0, 0.0, 0.0])
                was_moving = True
            elif args.execute and was_moving:
                # Do not leave the most recent velocity running after key
                # repeat ends, focus changes, or the watchdog fires.
                robot.stop()
                was_moving = False

            stdscr.erase()
            stdscr.addstr(0, 0, "ARM1 BASE-FRAME Y PROBE" +
                          ("  [ARMED]" if args.execute else "  [DRY RUN]"))
            stdscr.addstr(2, 0, "Hold W: +Y     Hold S: -Y     Space: STOP     Q: quit")
            stdscr.addstr(3, 0, "Only [0, +/-Y, 0] is permitted; no zigzag, policy, X/Z, or rotation.")
            stdscr.addstr(5, 0, "speed: %.1f mm/s    watchdog: %.0f ms" %
                          (args.speed * 1000.0, args.key_timeout * 1000.0))
            state = "+Y" if active > 0 else "-Y" if active < 0 else "STOPPED"
            stdscr.addstr(6, 0, "command: %s  velocity=%s" %
                          (state, np.array2string(velocity, precision=4)))
            stdscr.addstr(8, 0, _pose_text(robot))
            stdscr.addstr(10, 0, "Keep the physical E-stop accessible. Start with a brief tap.")
            stdscr.refresh()
            time.sleep(max(0.0, period - (time.monotonic() - loop_start)))
    finally:
        if args.execute:
            robot.stop()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.speed <= 0 or args.max_speed <= 0 or args.control_hz <= 0 or args.key_timeout <= 0:
        raise ValueError("speed, max-speed, control-hz, and key-timeout must be positive")
    if args.speed > args.max_speed:
        raise ValueError("--speed exceeds the hard --max-speed ceiling")

    # Keep imports deferred so --help and pure tests work off-robot.
    import rospy
    from bimanual_dressing.control.client import FrankaRobotClient

    rospy.init_node("drim_axis_probe", anonymous=True)
    robot = FrankaRobotClient(robot_name="arm1")
    if args.execute:
        input("Probe will command only arm1 +/-Y at %.1f mm/s. "
              "Verify E-stop, then press Enter to arm. " % (args.speed * 1000.0))
    else:
        print("Dry run: no robot command will be sent. Add --execute to arm.")
    try:
        return curses.wrapper(_ui, robot, args)
    except KeyboardInterrupt:
        return 0
    finally:
        if args.execute:
            robot.stop()


if __name__ == "__main__":
    raise SystemExit(main())
