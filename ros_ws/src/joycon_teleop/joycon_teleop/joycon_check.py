import argparse
import time

from joycon_teleop.joycon_lib import GyroTrackingJoyCon, get_L_id, get_R_id


def _pick_id(side: str):
    if side == "left":
        return get_L_id()
    if side == "right":
        return get_R_id()
    left_id = get_L_id()
    return left_id if None not in left_id else get_R_id()


def main():
    parser = argparse.ArgumentParser(description="Check Joy-Con connection and status.")
    parser.add_argument(
        "--side",
        choices=["left", "right", "auto"],
        default="auto",
        help="Which Joy-Con to use (default: auto).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Seconds to monitor for stability (default: 10).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Seconds between status reads (default: 0.5).",
    )
    args = parser.parse_args()

    joycon_id = _pick_id(args.side)
    if None in joycon_id:
        raise SystemExit("No Joy-Con detected (get_L_id/get_R_id returned None).")

    joycon = GyroTrackingJoyCon(*joycon_id)
    print(f"Connected: {joycon_id}")
    joycon.calibrate(seconds=2)
    time.sleep(2.5)

    start = time.time()
    while time.time() - start < args.duration:
        status = joycon.get_status()
        battery = status.get("battery", {})
        print(f"ok battery={battery.get('level')} charging={battery.get('charging')}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
