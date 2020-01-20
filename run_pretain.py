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
    "electra_config_file", None,
    "The config json file corresponding to the pre-trained ALBERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string('data_dir', None, "Path to data directory.")

flags.DEFINE_string('output_dir', 'results', "Path to output directory.")

flags.DEFINE_integer('max_seq_length', 128, "The maximum total input sequence length after WordPiece tokenization.")

flags.DEFINE_string('vocab_file', 'vocab.txt', "Path to vocabulary file.")

flags.DEFINE_boolean('do_lower_case', True, "Whether do lower case.")

flags.DEFINE_float('mask_percentage', 0.15, "Percentage of words to be masked for the generator.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


def get_config():
    config = ElectraConfig(30522)
    return config


def create_generator(config, is_training, input_ids):
    generator = Generator(config, is_training, input_ids)
    return generator


def create_discriminator(config, is_training, input_ids):
    discriminator = Discriminator(config, is_training, input_ids)
    return discriminator

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
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    electra_config = get_config()

    tf.gfile.MakeDirs(FLAGS.output_dir)



    data_dir = FLAGS.data_dir
    data = read_data(data_dir)
    print(data)


    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    input_ids = tokenize(data, tokenizer)
    print(len(input_ids))
    if len(input_ids) % FLAGS.max_seq_length:
        for i in range(FLAGS.max_seq_length - len(input_ids) % FLAGS.max_seq_length):
            input_ids.append(0)

    print(input_ids)
    input_ids = np.array(input_ids).reshape(int(len(input_ids)/FLAGS.max_seq_length), -1)
    print(input_ids[:4].shape)

    input_ids = tf.convert_to_tensor(input_ids)


    create_generator(config=config, is_training=True, input_ids=input_ids)
    model_summary()

    create_discriminator(config=config, is_training=True, input_ids=input_ids)
    model_summary()







if __name__ == '__main__':
    main()