import glob
import tensorflow as tf
import modeling
import optimization
import os
import sys
import random
import tokenization

flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "electra_config_file", None,
    "The config json file corresponding to the pre-trained ALBERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string('output_dir', None, "Path to output directory.")

flags.DEFINE_integer('max_seq_length', 128, "The maximum total input sequence length after WordPiece tokenization.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 39,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string('vocab_file', 'vocab.txt', "Path to vocabulary file.")

flags.DEFINE_boolean('do_lower_case', True, "Whether do lower case.")

flags.DEFINE_float('mask_percentage', 0.15, "Percentage of words to be masked for the generator.")

flags.DEFINE_float("learning_rate", 5e-4, "The initial learning rate for glue.")

flags.DEFINE_integer("num_train_steps", 10, "Number of training steps.")

#10000
flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 128, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

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

masked_token = ["[MASK]"]
tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
MASK_ID = tokenizer.convert_tokens_to_ids(masked_token)[0]


def get_config():
    config = modeling.ElectraConfig(30522)
    return config


def get_masked_lm_output(electra_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("generator"):
        with tf.variable_scope("cls/predictions"):
            # We apply one more non-linear transformation before the output layer.
            # This matrix is not used after pre-training.
            with tf.variable_scope("transform"):
                input_tensor = tf.layers.dense(
                    input_tensor,
                    units=electra_config.embedding_size,
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
            loss = tf.reduce_sum(label_weights * per_example_loss)
            #denominator = tf.reduce_sum(label_weights) + 1e-5
            #loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def get_discriminator_output(electra_config, sequence_tensor, whether_replaced, label_weights):
    label_weights = tf.cast(label_weights, dtype=tf.float32)

    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    sequence_tensor = tf.reshape(sequence_tensor, [batch_size * seq_length, width])

    with tf.variable_scope("discriminator"):
        with tf.variable_scope("whether_replaced/predictions"):
            output = tf.layers.dense(sequence_tensor,
                                     units=2,
                                     activation=modeling.get_activation(electra_config.hidden_act),
                                     kernel_initializer=modeling.create_initializer(
                                         electra_config.initializer_range))
            logits = modeling.layer_norm(output)

            '''
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            
            label_weights = tf.reshape(label_weights, [-1])

            one_hot_labels = tf.one_hot(whether_replaced, depth=2, dtype=tf.float32)

            per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
            numerator = tf.reduce_sum(label_weights * per_example_loss)
            denominator = tf.reduce_sum(label_weights) + 1e-5
            loss = numerator / denominator
            '''
            whether_replaced = tf.reshape(whether_replaced, [-1])
            one_hot_labels = tf.one_hot(whether_replaced, depth=2, dtype=tf.float32)

            print(one_hot_labels)
            print(logits)

            sigmoid_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=one_hot_labels,
                logits=logits,
                name='sigmoid_cross_entropy',
            )

            loss = tf.reduce_sum(sigmoid_cross_entropy)

    return (loss)


def replace_elements_by_indices(old, new, indices):
    old_shape = modeling.get_shape_list(old)
    batch_size = old_shape[0]
    seq_length = old_shape[1]


    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(indices + flat_offsets, [-1])

    zeros = tf.zeros(tf.shape(flat_positions)[0], dtype=tf.int32)

    flat_old = tf.reshape(old, [-1])

    masked_lm_mask = tf.sparse_to_dense(flat_positions, tf.shape(flat_old), zeros, default_value=1,
                                        validate_indices=True, name="masked_lm_mask")

    flat_old_temp = tf.multiply(flat_old, masked_lm_mask)
    new_temp = tf.sparse_to_dense(flat_positions, tf.shape(flat_old), new,
                                                        default_value=0, validate_indices=True, name=None)

    updated_old = tf.reshape(flat_old_temp + new_temp, old_shape)

    return updated_old




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

        batch_size = modeling.get_shape_list(input_ids)[0]
        masked_lm_positions = tf.constant([sorted(random.sample(range(0, FLAGS.max_seq_length), FLAGS.max_predictions_per_seq)) for i in range(batch_size)])
        masks_list = tf.constant([MASK_ID] * (FLAGS.max_predictions_per_seq * batch_size))
        masked_lm_weights = tf.ones(modeling.get_shape_list(masked_lm_positions))

        masked_input_ids = replace_elements_by_indices(input_ids, masks_list, masked_lm_positions)
        masked_lm_ids = gather_indexes_rank2(input_ids, masked_lm_positions)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        generator = modeling.Generator(config=electra_config,
                                     is_training=is_training,
                                     input_ids=masked_input_ids,
                                     input_mask=input_mask,
                                     use_one_hot_embeddings=use_one_hot_embeddings)

        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
         electra_config, generator.get_sequence_output(), generator.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights)

        masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)

        #zero = tf.constant(0, dtype=tf.int32)
        #positions_col2 = tf.reshape(masked_lm_positions, [-1])
        #non_zeros_coords = tf.where(tf.not_equal(positions_col2, zero))
        #print(modeling.get_shape_list(non_zeros_coords))

        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
        diff = masked_lm_predictions - masked_lm_ids


        #diff_cast = tf.gather_nd(tf.cast(tf.not_equal(diff, zero), dtype=tf.int32), non_zeros_coords)

        #index = tf.expand_dims(tf.range(0, batch_size), 1)
        #dup_index = tf.expand_dims(tf.reshape(tf.tile(index, multiples=[1, FLAGS.max_predictions_per_seq]), [-1]), 1)
        #positions = tf.concat([dup_index, tf.expand_dims(positions_col2, 1)], 1)
        #positions = tf.gather_nd(positions, non_zeros_coords)

        #whether_replaced = tf.sparse_to_dense(positions, tf.shape(input_ids), diff_cast, default_value=0,
                                              #validate_indices=True, name="whether_replaced")

        zeros = tf.zeros(modeling.get_shape_list(input_ids), dtype=tf.int32)
        whether_replaced = replace_elements_by_indices(zeros, diff, masked_lm_positions)


        #zeros = tf.zeros(tf.shape(non_zeros_coords)[0], dtype=tf.int32)
        #masked_lm_mask = tf.sparse_to_dense(positions, tf.shape(input_ids), zeros, default_value=1,
        #                                    validate_indices=True, name="masked_lm_mask")

        #input_ids_temp = tf.multiply(input_ids, masked_lm_mask)

        #masked_lm_predictions_positive = tf.gather_nd(masked_lm_predictions, non_zeros_coords)
        #masked_lm_predictions_temp = tf.sparse_to_dense(positions, tf.shape(input_ids), masked_lm_predictions_positive,
        #                                                default_value=0, validate_indices=True, name=None)

        input_ids_for_discriminator = replace_elements_by_indices(masked_input_ids, masked_lm_predictions, masked_lm_positions)

        discriminator = modeling.Discriminator(config=electra_config,
                                               is_training=is_training,
                                               input_ids=input_ids_for_discriminator,
                                               input_mask=input_mask,
                                               use_one_hot_embeddings=use_one_hot_embeddings)

        (disc_loss) = get_discriminator_output(electra_config, discriminator.get_sequence_output(),
                                                    whether_replaced, input_mask)


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

        """
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
        """
        output_spec = None
        total_loss = masked_lm_loss + disc_loss
        if mode == tf.estimator.ModeKeys.TRAIN:
            gen_train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu, "generator")

            disc_train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu, "discriminator")

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                #loss=masked_lm_loss + disc_loss,
                loss=total_loss,
                train_op=tf.group(gen_train_op, disc_train_op),
                scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = {
            "input_ids":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask":
                tf.FixedLenFeature([max_seq_length], tf.int64),
        }


        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_files))

            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            d = d.apply(
                tf.contrib.data.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=is_training,
                    cycle_length=cycle_length))
            d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)
            # Since we evaluate for a fixed number of steps we don't want to encounter
            # out-of-range exceptions.
            d = d.repeat()

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don't* want to drop the remainder, otherwise we wont cover
        # every sample.
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_batches=num_cpu_threads,
                drop_remainder=True))
        return d

    return input_fn


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


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


def gather_indexes_rank2(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=2)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  output_tensor = tf.reshape(output_tensor, [batch_size, FLAGS.max_predictions_per_seq])
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


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    electra_config = get_config()

    tf.gfile.MakeDirs(FLAGS.output_dir)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Input Files ***")
    for input_file in input_files:
        tf.logging.info("  %s" % input_file)


    #data_dir = FLAGS.data_dir
    #data = read_data(data_dir)
    #print(data)


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
        electra_config=electra_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    print("finish building model")

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size)

    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        train_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=True)
        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)



if __name__ == '__main__':
    main()