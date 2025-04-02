import sys
import argparse

from ice_station_zebra.commands import test, train

def main() -> None:
    # Select command
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["test", "train"], type=str)
    args, unknown = parser.parse_known_args()

    # Reset system arguments and run the appropriate hydra entrypoint
    sys.argv = [sys.argv[0]] + unknown
    if args.command == "test":
        test()
    elif args.command == "train":
        train()
