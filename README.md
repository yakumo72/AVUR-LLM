# Robust LLM-based Audio-Visual Speech Recognition with Sparse Modality Alignment and Visual Unit-Guided Refinement

This repository provides the code and data for the paper **"Robust LLM-based Audio-Visual Speech Recognition with Sparse Modality Alignment and Visual Unit-Guided Refinement"**.

## Overview

Our framework combines a pretrained **Whisper** audio encoder, an **AV-HuBERT** visual encoder, and a large language model (LLM) for robust audio-visual speech recognition.


## Installation

Create a fresh virtual environment:

```bash
conda create -n avur_llm python=3.10 -y
conda activate avur_llm
conda install conda-forge::sox
git clone https://github.com/anonymous42277/anonymous001.git
cd anonymous001
pip install torch==2.2.2+cu121 torchaudio==2.2.2+cu121 torchvision==0.17.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
cd ssl_models/av_hubert/fairseq
pip install --editable ./
```

## Checkpoints

Please download the required pretrained models and place them in the corresponding directories.

### Required pretrained models

1. **AV-HuBERT pretrained model**

   Download the AV-HuBERT pretrained model `large_noise_pt_noise_ft_433h.pt` from [here](http://facebookresearch.github.io/av_hubert) and place it at:

   ```text
   checkpoints/avhubert/large_noise_pt_noise_ft_433h.pt
   ```

2. **Whisper-medium model**

   Download the Whisper-medium model and place it under:

   ```text
   checkpoints/whisper_medium/
   ```

### Released checkpoints

The available checkpoints are listed below:

| Method | Module | Download Link |
|--------|--------|---------------|
| Whisper | Whisper-medium | https://pan.baidu.com/s/1bSpqzMYBaPx5v2AiJm8gIw?pwd=6a4h |
| AVUR-LLM | SMA | https://pan.baidu.com/s/1jbeqk507Z0CMHMsZT88CAw?pwd=ugcc |
| AVUR-LLM | SMA+AMF | https://pan.baidu.com/s/1-HH_wM0tS8RASXty9Pi-pA?pwd=a39w |
| AVUR-LLM | VUR | https://pan.baidu.com/s/12Kw9qaozKyGVdjRoniIvqg?pwd=pnps |

## Data Preparation

Please follow the [preprocessing](https://github.com/facebookresearch/av_hubert/blob/main/avhubert/preparation) guide to preprocess the LRS3 dataset.

## Inference

### Step 1. Prepare the test audio/video `data.list`

Prepare the test audio and video manifests. The `data.list` format is like:

```json
{"key": "test/Ip2SQa50uBI/00005", "wav": "audio/test/Ip2SQa50uBI/00005.wav", "txt": "but some of our grandchildren probably will"}
{"key": "test/Ip2SQa50uBI/00005", "wav": "video/test/Ip2SQa50uBI/00005.mp4", "txt": "but some of our grandchildren probably will"}
```

### Step 2. Run AVSR inference with SMA / SMA+AMF

Use the SMA or SMA+AMF model for AVSR inference. This stage can also generate N-best hypotheses, which will later be used by the VUR model.

You can directly run:

```bash
bash run_decode.sh
```

The core inference command is:

```bash
python infer_avsr.py \
  --gpu 0 \
  --config path/to/config.yaml \
  --data_type raw \
  --test_audio_data path/to/audio_data.list \
  --test_video_data path/to/video_data.list \
  --checkpoint path/to/checkpoint \
  --result_dir out_put
```

#### Argument description

- `--gpu`: GPU device ID used for inference.
- `--config`: Path to the inference configuration file.
- `--data_type`: Input data format. Use `raw` for raw audio/video paths listed in `data.list`.
- `--test_audio_data`: Path to the audio manifest file.
- `--test_video_data`: Path to the video manifest file.
- `--checkpoint`: Path to the checkpoint.
- `--result_dir`: Directory used to save decoding outputs.

### Step 3. Prepare visual units

Please follow the [clustering](https://github.com/facebookresearch/av_hubert/blob/main/avhubert/clustering) guide to prepare visual units.

In our experiments, we use:

- intermediate-layer features
- `K = 2000` clusters

The generated visual units, together with the N-best hypotheses from Step 2, are used as the input to the VUR model. The format of `units_nbest.list`:

```bash
{"key": "test/Ip2SQa50uBI/00005", "unit_ids": [1590, 142, 232, 232, 1408, 373, 10, 1725, ...], "candidates": [{"text": "they told our grandchildren probably as well", "s_main": -5.2917890548706055, "wer": 0.7142857142857143}, ...]}
.....
```
### Step 4. Run inference with the VUR model

After preparing the visual units and N-best hypotheses, run VUR inference:

```bash
bash run_decode.sh
```

The core command is:

```bash
python infer_llmavsr.py \
  --gpu 0 \
  --config path/to/config.yaml \
  --test_unit_nbest_data path/to/units_nbest.list \
  --checkpoint path/to/checkpoint.pt \
  --result_dir output
```

#### Argument description

- `--gpu`: GPU device ID used for inference.
- `--config`: Path to the inference configuration file.
- `--test_unit_nbest_data`: Path to the input list containing visual units and N-best hypotheses.
- `--checkpoint`: Path to the checkpoint.
- `--result_dir`: Directory used to save VUR decoding results.