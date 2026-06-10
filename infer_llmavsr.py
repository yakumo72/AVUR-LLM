import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from llmavsr_core import run_llmavsr_inference


def get_args():
    parser = argparse.ArgumentParser(description="AVUR-LLM inference")

    parser.add_argument("--config", required=True, help="config.yaml")
    parser.add_argument("--checkpoint", required=True, help="model checkpoint")

    parser.add_argument("--test_unit_nbest_data", required=True)

    parser.add_argument("--result_dir", required=True)

    parser.add_argument("--gpu", type=int, default=0)

    return parser.parse_args()


def main():
    args = get_args()
    run_llmavsr_inference(args)


if __name__ == "__main__":
    main()