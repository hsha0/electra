import glob
import tensorflow as tf
import os
import tokenization
import numpy as np
from model import *

flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "albert_config_file", None,
    "The config json file corresponding to the pre-trained ALBERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string('data_dir', None, "Path to data directory.")

flags.DEFINE_integer('max_seq_length', 128, "The maximum total input sequence length after WordPiece tokenization.")

flags.DEFINE_integer('dff', 2048, "The inner dimension of feed-forward network")

flags.DEFINE_integer('d_model', 512, "the number of expected features in the encoder/decoder inputs (default=512)")

flags.DEFINE_integer('gen_num_layers', 3, "Number of layers of the generator.")

flags.DEFINE_integer('dis_num_layers', 6, "Number of layers of the discriminator.")

flags.DEFINE_string('vocab_file', 'vocab.txt', "Path to vocabulary file.")

flags.DEFINE_boolean('do_lower_case', True, "Whether do lower case.")

flags.DEFINE_float('mask_percentage', 0.15, "Percentage of words to be masked for the generator.")

def get_config():
    config = ElectraConfig(30522)
    return config

def create_generator(config, is_training, input_ids):
    generator = Generator(config, is_training, input_ids)
    print(generator.get_sequence_output())

def model_fn_builder(config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, optimizer, poly_power,
                     start_warmup_step):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]



def read_data(data_dir):
    pre = os.getcwd()
    os.chdir(data_dir)
    data = ''
    for file in glob.glob('*.txt')[:1]:
        with open(file, 'r') as f:
            lines = [x for x in f.read().splitlines() if x != '']
            lines = ' '.join(lines)
            data += lines + ' '
    os.chdir(pre)

    return data


def tokenize(string, tokenizer):
    tokens = tokenizer.tokenize(string)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    return input_ids


def main():
    data_dir = FLAGS.data_dir
    data = read_data(data_dir)
    print(data)


    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)


    input_ids = tokenize(data, tokenizer)
    print(len(input_ids))
    if len(input_ids) % FLAGS.max_seq_length:
        for i in range(FLAGS.max_seq_length - len(input_ids) % FLAGS.max_seq_length):
            input_ids.append(0)

    print(input_ids)
    input_ids = np.array(input_ids).reshape(int(len(input_ids)/FLAGS.max_seq_length), -1)
    print(input_ids[:4].shape)

    input_ids = tf.convert_to_tensor(input_ids)

    config = get_config()
    create_generator(config=config, is_training=True, input_ids=input_ids)
    model_summary()







if __name__ == '__main__':
    main()