import tensorflow as tf
import six
import math
import json
import copy
import numpy as np
import collections
import re


class ElectraConfig(object):
    """
    Configuration for ElectraModel.
    """

    def __init__(self,
                 vocab_size,
                 embedding_size=128,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_hidden_groups=1,
                 num_attention_heads=4,
                 generator_size=0.25,
                 intermediate_size=1024,
                 inner_group_num=1,
                 down_scale_factor=1,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=128,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """
        Constructs ElectraConfig
        :param vocab_size:
        :param embedding_size:
        :param hidden_size:
        :param num_hidden_layers:
        :param num_hidden_groups:
        :param num_attention_heads:
        :param intermediate_size:
        :param inner_group_num:
        :param down_scale_factor:
        :param hidden_act:
        :param hidden_dropout_prob:
        :param attention_probs_dropout_prob:
        :param max_position_embeddings:
        :param type_vocab_size:
        :param initializer_range:
        """
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_groups = num_hidden_groups
        self.num_attention_heads = num_attention_heads
        self.generator_size = generator_size
        self.inner_group_num = inner_group_num
        self.down_scale_factor = down_scale_factor
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `ElectraConfig` from a Python dictionary of parameters."""
        config = ElectraConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `ElectraConfig` from a json file of parameters."""
        with tf.io.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class Generator(object):
    """
    Generator
    """

    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=False,
                 scope=None):
        """
        Constructor for ElectraModel.

        :param config:
        :param is_training:
        :param input_ids:
        :param input_mask:
        :param token_type_ids:
        :param use_one_hot_embeddings:
        :param scope:
        """
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        with tf.compat.v1.variable_scope("embeddings"):
            # Perform embedding lookup on the word ids.
            (self.embedding_output, self.embedding_table) = embedding_lookup(
                input_ids=input_ids,
                vocab_size=config.vocab_size,
                embedding_size=config.embedding_size,
                initializer_range=config.initializer_range,
                word_embedding_name="word_embeddings",
                use_one_hot_embeddings=use_one_hot_embeddings)

            # Add positional embeddings and token type embeddings, then layer
            # normalize and perform dropout.
            self.embedding_output = embedding_postprocessor(
                input_tensor=self.embedding_output,
                use_token_type=True,
                token_type_ids=token_type_ids,
                token_type_vocab_size=config.type_vocab_size,
                token_type_embedding_name="token_type_embeddings",
                use_position_embeddings=True,
                position_embedding_name="position_embeddings",
                initializer_range=config.initializer_range,
                max_position_embeddings=config.max_position_embeddings,
                dropout_prob=config.hidden_dropout_prob)

        with tf.compat.v1.variable_scope(scope, default_name="generator"):
            with tf.compat.v1.variable_scope("encoder"):
                self.all_encoder_layers = transformer_model(
                    input_tensor=self.embedding_output,
                    attention_mask=input_mask,
                    hidden_size=int(config.hidden_size * config.generator_size),
                    num_hidden_layers=config.num_hidden_layers,
                    num_hidden_groups=config.num_hidden_groups,
                    num_attention_heads=int(config.num_attention_heads * config.generator_size),
                    intermediate_size=int(config.intermediate_size * config.generator_size),
                    inner_group_num=config.inner_group_num,
                    intermediate_act_fn=get_activation(config.hidden_act),
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    initializer_range=config.initializer_range,
                    do_return_all_layers=True)

            self.sequence_output = self.all_encoder_layers[-1]

    def get_sequence_output(self):
        """Gets final hidden layer of encoder.
            Returns:
              float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
              to the final hidden of the transformer encoder.
            """
        return self.sequence_output

    def get_embedding_table(self):
        return self.embedding_table


class Discriminator(object):
    """
    Discriminator
    """

    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 train_pooler=True,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=False,
                 scope=None):
        """
        Constructor for ElectraModel.

        :param config:
        :param is_training:
        :param input_ids:
        :param input_mask:
        :param token_type_ids:
        :param use_one_hot_embeddings:
        :param scope:
        """
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        with tf.compat.v1.variable_scope("embeddings", reuse=tf.compat.v1.AUTO_REUSE):
            # Perform embedding lookup on the word ids.
            (self.embedding_output, self.embedding_table) = embedding_lookup(
                input_ids=input_ids,
                vocab_size=config.vocab_size,
                embedding_size=config.embedding_size,
                initializer_range=config.initializer_range,
                word_embedding_name="word_embeddings",
                use_one_hot_embeddings=use_one_hot_embeddings)

            # Add positional embeddings and token type embeddings, then layer
            # normalize and perform dropout.
            self.embedding_output = embedding_postprocessor(
                input_tensor=self.embedding_output,
                use_token_type=True,
                token_type_ids=token_type_ids,
                token_type_vocab_size=config.type_vocab_size,
                token_type_embedding_name="token_type_embeddings",
                use_position_embeddings=True,
                position_embedding_name="position_embeddings",
                initializer_range=config.initializer_range,
                max_position_embeddings=config.max_position_embeddings,
                dropout_prob=config.hidden_dropout_prob)

        with tf.compat.v1.variable_scope(scope, default_name="discriminator"):
            with tf.compat.v1.variable_scope("encoder"):
                self.all_encoder_layers = transformer_model(
                    input_tensor=self.embedding_output,
                    attention_mask=input_mask,
                    hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers,
                    num_hidden_groups=config.num_hidden_groups,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    inner_group_num=config.inner_group_num,
                    intermediate_act_fn=get_activation(config.hidden_act),
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    initializer_range=config.initializer_range,
                    do_return_all_layers=True)

            self.sequence_output = self.all_encoder_layers[-1]

            with tf.compat.v1.variable_scope("pooler"):
                # We "pool" the model by simply taking the hidden state corresponding
                # to the first token. We assume that this has been pre-trained
                first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
                self.pooled_output = tf.compat.v1.layers.dense(
                    first_token_tensor,
                    config.hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=create_initializer(config.initializer_range),
                    trainable=train_pooler,
                )

    def get_sequence_output(self):
        """Gets final hidden layer of encoder.
            Returns:
              float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
              to the final hidden of the transformer encoder.
            """
        return self.sequence_output

    def get_pooled_output(self):
        return self.pooled_output

    def get_embedding_table(self):
        return self.embedding_table


def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.
      Args:
        activation_string: String name of the activation function.
      Returns:
        A Python function corresponding to the activation function. If
        `activation_string` is None, empty, or "linear", this will return None.
        If `activation_string` is not a string, it will return `activation_string`.
      Raises:
        ValueError: The `activation_string` does not correspond to a known
          activation.
    """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)


def gelu(x):
    """Gaussian Error Linear Unit.
      This is a smoother version of the RELU.
      Original paper: https://arxiv.org/abs/1606.08415
      Args:
        x: float Tensor to perform activation.
      Returns:
        `x` with the GELU activation applied.
      """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.compat.v1.truncated_normal_initializer(stddev=initializer_range)


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.compat.v1.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(input=tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12, name=name)(input_tensor)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    #assignment_map[name] = name
    assignment_map[name] = name_to_variable[name]
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, rate=(1 - (1.0 - dropout_prob)))
    return output


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
    """Runs layer normalization followed by dropout."""
    output_tensor = layer_norm(input_tensor, name)
    output_tensor = dropout(output_tensor, dropout_prob)
    return output_tensor


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
    """
    Looks up words embeddings for id tensor.
    :param input_ids:
    :param vocab_size:
    :param embedding_size:
    :param initializer_range:
    :param word_embedding_name:
    :param use_one_hot_embeddings:
    :return:
    """

    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])

    embedding_table = tf.compat.v1.get_variable(
        name=word_embedding_name,
        shape=[vocab_size, embedding_size],
        initializer=create_initializer(initializer_range)
    )

    flat_input_ids = tf.reshape(input_ids, [-1])
    if use_one_hot_embeddings:
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
        output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
        output = tf.gather(embedding_table, flat_input_ids)

    input_shape = get_shape_list(input_ids)

    output = tf.reshape(output,
                        input_shape[0:-1] + [input_shape[-1] * embedding_size])

    return (output, embedding_table)


def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
    """
    Performs position embeddings and segment embeddings.
    :param input_tensor:
    :param use_token_type:
    :param token_type_ids:
    :param token_type_vocab_size:
    :param token_type_embedding_name:
    :param use_position_embeddings:
    :param position_embedding_name:
    :param initializer_range:
    :param max_position_embeddings:
    :param dropout_prob:
    :return:
    """

    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = input_tensor

    if use_token_type:
        if token_type_ids is None:
            raise ValueError("`token_type_ids` must be specified if"
                             "`use_token_type` is True.")
        token_type_table = tf.compat.v1.get_variable(
            name=token_type_embedding_name,
            shape=[token_type_vocab_size, width],
            initializer=create_initializer(initializer_range)
        )

        flat_token_type_ids = tf.reshape(token_type_ids, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
        token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
        token_type_embeddings = tf.reshape(token_type_embeddings,
                                           [batch_size, seq_length, width])
        output += token_type_embeddings

    if use_position_embeddings:
        assert_op = tf.compat.v1.assert_less_equal(seq_length, max_position_embeddings)
        with tf.control_dependencies([assert_op]):
            full_position_embeddings = tf.compat.v1.get_variable(
                name=position_embedding_name,
                shape=[max_position_embeddings, width],
                initializer=create_initializer(initializer_range))
        position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                       [seq_length, -1])
        num_dims = len(output.shape.as_list())

        position_broadcast_shape = []
        for _ in range(num_dims - 2):
            position_broadcast_shape.append(1)
        position_broadcast_shape.extend([seq_length, width])
        position_embeddings = tf.reshape(position_embeddings,
                                         position_broadcast_shape)
        output += position_embeddings

    output = layer_norm_and_dropout(output, dropout_prob)
    return output


def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_hidden_groups=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      inner_group_num=1,
                      intermediate_act_fn="gelu",
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):

    """Multi-headed, multi-layer Transformer from "Attention is All You Need".
      This is almost an exact implementation of the original Transformer encoder.
      See the original paper:
      https://arxiv.org/abs/1706.03762
      Also see:
      https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
      Args:
        input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
        attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
          seq_length], with 1 for positions that can be attended to and 0 in
          positions that should not be.
        hidden_size: int. Hidden size of the Transformer.
        num_hidden_layers: int. Number of layers (blocks) in the Transformer.
        num_hidden_groups: int. Number of group for the hidden layers, parameters
          in the same group are shared.
        num_attention_heads: int. Number of attention heads in the Transformer.
        intermediate_size: int. The size of the "intermediate" (a.k.a., feed
          forward) layer.
        inner_group_num: int, number of inner repetition of attention and ffn.
        intermediate_act_fn: function. The non-linear activation function to apply
          to the output of the intermediate/feed-forward layer.
        hidden_dropout_prob: float. Dropout probability for the hidden layers.
        attention_probs_dropout_prob: float. Dropout probability of the attention
          probabilities.
        initializer_range: float. Range of the initializer (stddev of truncated
          normal).
        do_return_all_layers: Whether to also return all layers or just the final
          layer.
      Returns:
        float Tensor of shape [batch_size, seq_length, hidden_size], the final
        hidden layer of the Transformer.
      Raises:
        ValueError: A Tensor shape or parameter is invalid.
    """

    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))

    attention_head_size = hidden_size // num_attention_heads
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    input_width = input_shape[2]

    all_layer_outputs = []
    if input_width != hidden_size:
        prev_output = dense_layer_2d(input_tensor, hidden_size, create_initializer(initializer_range),
                                     None, name="embedding_hidden_mapping_in")
    else:
        prev_output = input_tensor
    with tf.compat.v1.variable_scope("transformer", reuse=tf.compat.v1.AUTO_REUSE):
        for layer_idx in range(num_hidden_layers):
            group_idx = int(layer_idx / num_hidden_layers * num_hidden_groups)
            with tf.compat.v1.variable_scope("group_%d" % group_idx):
                with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
                    layer_output = prev_output
                    for inner_group_idx in range(inner_group_num):
                        with tf.compat.v1.variable_scope("inner_group_%d" % inner_group_idx):
                            layer_output = attention_ffn_block(
                                layer_output, hidden_size, attention_mask,
                                num_attention_heads, attention_head_size,
                                attention_probs_dropout_prob, intermediate_size,
                                intermediate_act_fn, initializer_range, hidden_dropout_prob)
                            prev_output = layer_output
                            all_layer_outputs.append(layer_output)
    if do_return_all_layers:
        return all_layer_outputs
    else:
        return all_layer_outputs[-1]

def dense_layer_2d(input_tensor,
                   output_size,
                   initializer,
                   activation,
                   num_attention_heads=1,
                   name=None):

    """A dense layer with 2D kernel.
      Args:
        input_tensor: Float tensor with rank 3.
        output_size: The size of output dimension.
        initializer: Kernel initializer.
        activation: Activation function.
        num_attention_heads: number of attention head in attention layer.
        name: The name scope of this layer.
      Returns:
        float logits Tensor.
    """

    del num_attention_heads
    input_shape = get_shape_list(input_tensor)
    hidden_size = input_shape[2]

    with tf.compat.v1.variable_scope(name):
        w = tf.compat.v1.get_variable(
            name="kernel",
            shape=[hidden_size, output_size],
            initializer=initializer)
        b = tf.compat.v1.get_variable(
            name="bias",
            shape=[output_size],
            initializer=tf.compat.v1.zeros_initializer)
        ret = tf.einsum("BFH,HO->BFO", input_tensor, w)
        ret += b

    if activation is not None:
        return activation(ret)
    else:
        return ret


def attention_ffn_block(layer_input,
                        hidden_size=768,
                        attention_mask=None,
                        num_attention_heads=1,
                        attention_head_size=64,
                        attention_probs_dropout_prob=0.0,
                        intermediate_size=3072,
                        intermediate_act_fn=None,
                        initializer_range=0.02,
                        hidden_dropout_prob=0.0):
    """A network with attention-ffn as sub-block.
      Args:
        layer_input: float Tensor of shape [batch_size, from_seq_length,
          from_width].
        hidden_size: (optional) int, size of hidden layer.
        attention_mask: (optional) int32 Tensor of shape [batch_size,
          from_seq_length, to_seq_length]. The values should be 1 or 0. The
          attention scores will effectively be set to -infinity for any positions in
          the mask that are 0, and will be unchanged for positions that are 1.
        num_attention_heads: int. Number of attention heads.
        attention_head_size: int. Size of attention head.
        attention_probs_dropout_prob: float. dropout probability for attention_layer
        intermediate_size: int. Size of intermediate hidden layer.
        intermediate_act_fn: (optional) Activation function for the intermediate
          layer.
        initializer_range: float. Range of the weight initializer.
        hidden_dropout_prob: (optional) float. Dropout probability of the hidden
          layer.
      Returns:
        layer output
    """

    with tf.compat.v1.variable_scope("attention_1"):
        with tf.compat.v1.variable_scope("self"):
            attention_output = attention_layer(
                from_tensor=layer_input,
                to_tensor=layer_input,
                attention_mask=attention_mask,
                num_attention_heads=num_attention_heads,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                initializer_range=initializer_range)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.compat.v1.variable_scope("output"):
            attention_output = dense_layer_3d_proj(
                attention_output,
                hidden_size,
                attention_head_size,
                create_initializer(initializer_range),
                None,
                name="dense")
            attention_output = dropout(attention_output, hidden_dropout_prob)
        attention_output = layer_norm(attention_output + layer_input)
        with tf.compat.v1.variable_scope("ffn_1"):
            with tf.compat.v1.variable_scope("intermediate"):
                intermediate_output = dense_layer_2d(
                    attention_output,
                    intermediate_size,
                    create_initializer(initializer_range),
                    intermediate_act_fn,
                    num_attention_heads=num_attention_heads,
                    name="dense")
            with tf.compat.v1.variable_scope("output"):
                ffn_output = dense_layer_2d(
                    intermediate_output,
                    hidden_size,
                    create_initializer(initializer_range),
                    None,
                    num_attention_heads=num_attention_heads,
                    name="dense")
            ffn_output = dropout(ffn_output, hidden_dropout_prob)
        ffn_output = layer_norm(ffn_output + attention_output)
        return ffn_output


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
    """Performs multi-headed attention from `from_tensor` to `to_tensor`.
      Args:
        from_tensor: float Tensor of shape [batch_size, from_seq_length,
          from_width].
        to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
        attention_mask: (optional) int32 Tensor of shape [batch_size,
          from_seq_length, to_seq_length]. The values should be 1 or 0. The
          attention scores will effectively be set to -infinity for any positions in
          the mask that are 0, and will be unchanged for positions that are 1.
        num_attention_heads: int. Number of attention heads.
        query_act: (optional) Activation function for the query transform.
        key_act: (optional) Activation function for the key transform.
        value_act: (optional) Activation function for the value transform.
        attention_probs_dropout_prob: (optional) float. Dropout probability of the
          attention probabilities.
        initializer_range: float. Range of the weight initializer.
        batch_size: (Optional) int. If the input is 2D, this might be the batch size
          of the 3D version of the `from_tensor` and `to_tensor`.
        from_seq_length: (Optional) If the input is 2D, this might be the seq length
          of the 3D version of the `from_tensor`.
        to_seq_length: (Optional) If the input is 2D, this might be the seq length
          of the 3D version of the `to_tensor`.
      Returns:
        float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
          size_per_head].
      Raises:
        ValueError: Any of the arguments or tensor shapes are invalid.
    """

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])
    size_per_head = int(from_shape[2]/num_attention_heads)

    if len(from_shape) != len(to_shape):
        raise ValueError("The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError("When passing in rank 2 tensors to attention_layer, the values "
                             "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                             "musk all be specified.")

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    # `query_layer` = [B, F, N, H]
    q = dense_layer_3d(from_tensor, num_attention_heads, size_per_head,
                       create_initializer(initializer_range), query_act, "query")

    # `key_layer` = [B, T, N, H]
    k = dense_layer_3d(to_tensor, num_attention_heads, size_per_head,
                       create_initializer(initializer_range), key_act, "key")
    # `value_layer` = [B, T, N, H]
    v = dense_layer_3d(to_tensor, num_attention_heads, size_per_head,
                       create_initializer(initializer_range), value_act, "value")
    q = tf.transpose(a=q, perm=[0, 2, 1, 3])
    k = tf.transpose(a=k, perm=[0, 2, 1, 3])
    v = tf.transpose(a=v, perm=[0, 2, 1, 3])
    if attention_mask is not None:
        attention_mask = tf.reshape(
            attention_mask, [batch_size, 1, to_seq_length, 1])
        # 'new_embeddings = [B, N, F, H]'
    new_embeddings = dot_product_attention(q, k, v, attention_mask,
                                           attention_probs_dropout_prob)

    return tf.transpose(a=new_embeddings, perm=[0, 2, 1, 3])


def dense_layer_3d(input_tensor,
                   num_attention_heads,
                   head_size,
                   initializer,
                   activation,
                   name=None):
    """A dense layer with 3D kernel.
      Args:
        input_tensor: float Tensor of shape [batch, seq_length, hidden_size].
        num_attention_heads: Number of attention heads.
        head_size: The size per attention head.
        initializer: Kernel initializer.
        activation: Actication function.
        name: The name scope of this layer.
      Returns:
        float logits Tensor.
    """

    input_shape = get_shape_list(input_tensor)
    hidden_size = input_shape[2]

    with tf.compat.v1.variable_scope(name):
        w = tf.compat.v1.get_variable(
            name="kernel",
            shape=[hidden_size, num_attention_heads * head_size],
            initializer=initializer)
        w = tf.reshape(w, [hidden_size, num_attention_heads, head_size])
        b = tf.compat.v1.get_variable(
            name="bias",
            shape=[num_attention_heads * head_size],
            initializer=tf.compat.v1.zeros_initializer)
        b = tf.reshape(b, [num_attention_heads, head_size])
        ret = tf.einsum("BFH,HND->BFND", input_tensor, w)
        ret += b
    if activation is not None:
        return activation(ret)
    else:
        return ret


def dot_product_attention(q, k, v, bias, dropout_rate=0.0):
    """Dot-product attention.
      Args:
        q: Tensor with shape [..., length_q, depth_k].
        k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
          match with q.
        v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must
          match with q.
        bias: bias Tensor (see attention_bias())
        dropout_rate: a float.
      Returns:
        Tensor with shape [..., length_q, depth_v].
    """
    logits = tf.matmul(q, k, transpose_b=True)  # [..., length_q, length_kv]
    logits = tf.multiply(logits, 1.0 / math.sqrt(float(get_shape_list(q)[-1])))
    if bias is not None:
        # `attention_mask` = [B, T]
        from_shape = get_shape_list(q)
        if len(from_shape) == 4:
            broadcast_ones = tf.ones([from_shape[0], 1, from_shape[2], 1], tf.float32)
        elif len(from_shape) == 5:
            # from_shape = [B, N, Block_num, block_size, depth]#
            broadcast_ones = tf.ones([from_shape[0], 1, from_shape[2], from_shape[3],
                                      1], tf.float32)

        bias = tf.matmul(broadcast_ones,
                         tf.cast(bias, tf.float32), transpose_b=True)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - bias) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        logits += adder
    else:
        adder = 0.0

    attention_probs = tf.nn.softmax(logits, name="attention_probs")
    attention_probs = dropout(attention_probs, dropout_rate)
    return tf.matmul(attention_probs, v)


def dense_layer_3d_proj(input_tensor,
                        hidden_size,
                        head_size,
                        initializer,
                        activation,
                        name=None):
    """A dense layer with 3D kernel for projection.
      Args:
        input_tensor: float Tensor of shape [batch,from_seq_length,
          num_attention_heads, size_per_head].
        hidden_size: The size of hidden layer.
        num_attention_heads: The size of output dimension.
        head_size: The size of head.
        initializer: Kernel initializer.
        activation: Actication function.
        name: The name scope of this layer.
      Returns:
        float logits Tensor.
    """
    input_shape = get_shape_list(input_tensor)
    num_attention_heads = input_shape[2]
    with tf.compat.v1.variable_scope(name):
        w = tf.compat.v1.get_variable(
            name="kernel",
            shape=[num_attention_heads * head_size, hidden_size],
            initializer=initializer)
        w = tf.reshape(w, [num_attention_heads, head_size, hidden_size])
        b = tf.compat.v1.get_variable(
            name="bias", shape=[hidden_size], initializer=tf.compat.v1.zeros_initializer)
        ret = tf.einsum("BFND,NDH->BFH", input_tensor, w)
        ret += b
    if activation is not None:
        return activation(ret)
    else:
        return ret