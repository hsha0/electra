# ELECTRA

## Introduction
My own implementation of ELECTRA.

## Pretraining
```commandline
python3 run_pretrain.py \
--electra_config_file=CONFIG_FILE \
--input_file=INPUT_DIR/*.tfrecord \
--output_dir=OUTPUT_DIR \
--vocab_file= VOCAB_FILE \
--model=ale (one of [ale, electra]*) \
--do_train=True \
--learning_rate=5e-4 \
--train_batch_size=128 \
--max_seq_length=128 \
--num_train_steps=1000000 \
--max_predictions_per_seq=20 \
--save_checkpoints_steps=1000 \
--iterations_per_loop=1000 \
--use_tpu=true \
--tpu_name=TPU_NAME
```
\* ale: generator and discriminator will be built upon ALBERT instead of BERT.
