import argparse

import mujoco
import mujoco.viewer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuJoCo scene viewer.")
    parser.add_argument("scene", help="Absolute path to MuJoCo scene XML.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)


if __name__ == "__main__":
    main()
