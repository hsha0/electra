import glob
import tensorflow as tf
import os
import tokenization
import numpy as np
import modeling
import optimization

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
    config = modeling.ElectraConfig(30522)
    return config


def create_generator(config, is_training, input_ids):
    generator = modeling.Generator(config, is_training, input_ids)
    return generator


def create_discriminator(config, is_training, input_ids):
    discriminator = modeling.Discriminator(config, is_training, input_ids)
    return discriminator


def get_masked_lm_output(electra_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=electra_config.hidden_size,
                activation=modeling.get_activation(electra_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    electra_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[electra_config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=electra_config.vocab_size, dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)



def model_fn_builder(electra_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        generator = create_generator(config=electra_config,
                                     is_training=is_training,
                                     input_ids=input_ids,
                                     input_mask=input_mask,
                                     use_one_hot_embeddings=use_one_hot_embeddings)

        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
         electra_config, generator.get_sequence_output(), generator.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights)

        total_loss = masked_lm_loss

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


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

    model_fn = model_fn_builder(
        bert_config=electra_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    input_ids = tokenize(data, tokenizer)
    print(len(input_ids))
    if len(input_ids) % FLAGS.max_seq_length:
        for i in range(FLAGS.max_seq_length - len(input_ids) % FLAGS.max_seq_length):
            input_ids.append(0)

    print(input_ids)
    input_ids = np.array(input_ids).reshape(int(len(input_ids)/FLAGS.max_seq_length), -1)
    print(input_ids[:4].shape)

    input_ids = tf.convert_to_tensor(input_ids)







if __name__ == '__main__':
    main()