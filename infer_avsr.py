import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from avsr_core import run_avsr_inference


def get_args():
    parser = argparse.ArgumentParser(description="AVUR-LLM inference")

    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to decode checkpoint")
    parser.add_argument("--test_audio_data", required=True, help="Path to audio data.list")
    parser.add_argument("--test_video_data", required=True, help="Path to video data.list")
    parser.add_argument("--result_dir", required=True, help="Output result directory")

    parser.add_argument("--gpu", type=int, default=0, help="GPU id, default: 0")

    parser.add_argument(
        "--data_type",
        default="shard",
        choices=["raw", "shard"],
        help="Input data type",
    )

    parser.add_argument("--add_noise", action="store_true", default=False)
    parser.add_argument("--noise_snr_test", type=float, default=0.0)
    parser.add_argument("--noise_fn_test", type=str, default=None)

    return parser.parse_args()


def main():
    args = get_args()
    run_avsr_inference(args)


if __name__ == "__main__":
    main()