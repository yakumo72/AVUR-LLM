python infer_avsr.py \
  --gpu 0 \
  --config path/to/config.yaml \
  --data_type raw \
  --test_audio_data path/to/audio_data.list \
  --test_video_data path/to/video_data.list \
  --checkpoint path/to/checkpoint \
  --result_dir out_put


python infer_llmavsr.py \
  --gpu 0 \
  --config path/to/config.yaml \
  --test_unit_nbest_data path/to/units_nbest.list \
  --checkpoint path/to/checkpoint.pt \
  --result_dir output