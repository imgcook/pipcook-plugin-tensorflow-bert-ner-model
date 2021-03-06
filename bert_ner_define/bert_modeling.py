from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import math
import six
import tensorflow as tf

from .utils import tf_utils


class BertConfig(object):

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02,
               backward_compatible=True):
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range
    self.backward_compatible = backward_compatible

  @classmethod
  def from_dict(cls, json_object):
    config = BertConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    with tf.io.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def get_bert_model(input_word_ids,
                   input_mask,
                   input_type_ids,
                   config=None,
                   name=None,
                   float_type=tf.float32):
  bert_model_layer = BertModel(config=config, float_type=float_type, name=name)
  pooled_output, sequence_output = bert_model_layer(input_word_ids, input_mask,
                                                    input_type_ids)
  bert_model = tf.keras.Model(
      inputs=[input_word_ids, input_mask, input_type_ids],
      outputs=[pooled_output, sequence_output])
  return bert_model


class BertModel(tf.keras.layers.Layer):
  def __init__(self, config, float_type=tf.float32, **kwargs):
    super(BertModel, self).__init__(**kwargs)
    self.config = (
        BertConfig.from_dict(config)
        if isinstance(config, dict) else copy.deepcopy(config))
    self.float_type = float_type

  def build(self, unused_input_shapes):
    self.embedding_lookup = EmbeddingLookup(
        vocab_size=self.config.vocab_size,
        embedding_size=self.config.hidden_size,
        initializer_range=self.config.initializer_range,
        dtype=tf.float32,
        name="word_embeddings")
    self.embedding_postprocessor = EmbeddingPostprocessor(
        use_type_embeddings=True,
        token_type_vocab_size=self.config.type_vocab_size,
        use_position_embeddings=True,
        max_position_embeddings=self.config.max_position_embeddings,
        dropout_prob=self.config.hidden_dropout_prob,
        initializer_range=self.config.initializer_range,
        dtype=tf.float32,
        name="embedding_postprocessor")
    self.encoder = Transformer(
        num_hidden_layers=self.config.num_hidden_layers,
        hidden_size=self.config.hidden_size,
        num_attention_heads=self.config.num_attention_heads,
        intermediate_size=self.config.intermediate_size,
        intermediate_activation=self.config.hidden_act,
        hidden_dropout_prob=self.config.hidden_dropout_prob,
        attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
        initializer_range=self.config.initializer_range,
        backward_compatible=self.config.backward_compatible,
        float_type=self.float_type,
        name="encoder")
    self.pooler_transform = tf.keras.layers.Dense(
        units=self.config.hidden_size,
        activation="tanh",
        kernel_initializer=get_initializer(self.config.initializer_range),
        name="pooler_transform")
    super(BertModel, self).build(unused_input_shapes)

  def __call__(self,
               input_word_ids,
               input_mask=None,
               input_type_ids=None,
               **kwargs):
    inputs = tf_utils.pack_inputs([input_word_ids, input_mask, input_type_ids])
    return super(BertModel, self).__call__(inputs, **kwargs)

  def call(self, inputs, mode="bert", **kwargs):
    unpacked_inputs = tf_utils.unpack_inputs(inputs)
    input_word_ids = unpacked_inputs[0]
    input_mask = unpacked_inputs[1]
    input_type_ids = unpacked_inputs[2]

    word_embeddings = self.embedding_lookup(input_word_ids)
    embedding_tensor = self.embedding_postprocessor(
        word_embeddings=word_embeddings, token_type_ids=input_type_ids)
    if self.float_type == tf.float16:
      embedding_tensor = tf.cast(embedding_tensor, tf.float16)
    attention_mask = None
    if input_mask is not None:
      attention_mask = create_attention_mask_from_input_mask(
          input_word_ids, input_mask)

    if mode == "encoder":
      return self.encoder(
          embedding_tensor, attention_mask, return_all_layers=True)

    sequence_output = self.encoder(embedding_tensor, attention_mask)
    first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
    pooled_output = self.pooler_transform(first_token_tensor)

    return (pooled_output, sequence_output)

  def get_config(self):
    config = {"config": self.config.to_dict()}
    base_config = super(BertModel, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class EmbeddingLookup(tf.keras.layers.Layer):

  def __init__(self,
               vocab_size,
               embedding_size=768,
               initializer_range=0.02,
               **kwargs):
    super(EmbeddingLookup, self).__init__(**kwargs)
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.initializer_range = initializer_range

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.embeddings = self.add_weight(
        "embeddings",
        shape=[self.vocab_size, self.embedding_size],
        initializer=get_initializer(self.initializer_range),
        dtype=self.dtype)
    super(EmbeddingLookup, self).build(unused_input_shapes)

  def call(self, inputs):
    """Implements call() for the layer."""
    input_shape = tf_utils.get_shape_list(inputs)
    flat_input = tf.reshape(inputs, [-1])
    output = tf.gather(self.embeddings, flat_input)
    output = tf.reshape(output, input_shape + [self.embedding_size])
    return output


class EmbeddingPostprocessor(tf.keras.layers.Layer):

  def __init__(self,
               use_type_embeddings=False,
               token_type_vocab_size=None,
               use_position_embeddings=True,
               max_position_embeddings=512,
               dropout_prob=0.0,
               initializer_range=0.02,
               initializer=None,
               **kwargs):
    super(EmbeddingPostprocessor, self).__init__(**kwargs)
    self.use_type_embeddings = use_type_embeddings
    self.token_type_vocab_size = token_type_vocab_size
    self.use_position_embeddings = use_position_embeddings
    self.max_position_embeddings = max_position_embeddings
    self.dropout_prob = dropout_prob
    self.initializer_range = initializer_range

    if not initializer:
      self.initializer = get_initializer(self.initializer_range)
    else:
      self.initializer = initializer

    if self.use_type_embeddings and not self.token_type_vocab_size:
      raise ValueError("If `use_type_embeddings` is True, then "
                       "`token_type_vocab_size` must be specified.")

  def build(self, input_shapes):
    (word_embeddings_shape, _) = input_shapes
    width = word_embeddings_shape.as_list()[-1]
    self.type_embeddings = None
    if self.use_type_embeddings:
      self.type_embeddings = self.add_weight(
          "type_embeddings",
          shape=[self.token_type_vocab_size, width],
          initializer=get_initializer(self.initializer_range),
          dtype=self.dtype)

    self.position_embeddings = None
    if self.use_position_embeddings:
      self.position_embeddings = self.add_weight(
          "position_embeddings",
          shape=[self.max_position_embeddings, width],
          initializer=get_initializer(self.initializer_range),
          dtype=self.dtype)

    self.output_layer_norm = tf.keras.layers.LayerNormalization(
        name="layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32)
    self.output_dropout = tf.keras.layers.Dropout(rate=self.dropout_prob,
                                                  dtype=tf.float32)
    super(EmbeddingPostprocessor, self).build(input_shapes)

  def __call__(self, word_embeddings, token_type_ids=None, **kwargs):
    inputs = tf_utils.pack_inputs([word_embeddings, token_type_ids])
    return super(EmbeddingPostprocessor, self).__call__(inputs, **kwargs)

  def call(self, inputs, **kwargs):
    unpacked_inputs = tf_utils.unpack_inputs(inputs)
    word_embeddings = unpacked_inputs[0]
    token_type_ids = unpacked_inputs[1]
    input_shape = tf_utils.get_shape_list(word_embeddings, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = word_embeddings
    if self.use_type_embeddings:
      flat_token_type_ids = tf.reshape(token_type_ids, [-1])
      one_hot_ids = tf.one_hot(
          flat_token_type_ids,
          depth=self.token_type_vocab_size,
          dtype=self.dtype)
      token_type_embeddings = tf.matmul(one_hot_ids, self.type_embeddings)
      token_type_embeddings = tf.reshape(token_type_embeddings,
                                         [batch_size, seq_length, width])
      output += token_type_embeddings

    if self.use_position_embeddings:
      position_embeddings = tf.expand_dims(
          tf.slice(self.position_embeddings, [0, 0], [seq_length, width]),
          axis=0)

      output += position_embeddings

    output = self.output_layer_norm(output)
    output = self.output_dropout(output,training=kwargs.get('training', False))

    return output


class Attention(tf.keras.layers.Layer):

  def __init__(self,
               num_attention_heads=12,
               size_per_head=64,
               attention_probs_dropout_prob=0.0,
               initializer_range=0.02,
               backward_compatible=False,
               **kwargs):
    super(Attention, self).__init__(**kwargs)
    self.num_attention_heads = num_attention_heads
    self.size_per_head = size_per_head
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_range = initializer_range
    self.backward_compatible = backward_compatible

  def build(self, unused_input_shapes):
    self.query_dense = self._projection_dense_layer("query")
    self.key_dense = self._projection_dense_layer("key")
    self.value_dense = self._projection_dense_layer("value")
    self.attention_probs_dropout = tf.keras.layers.Dropout(
        rate=self.attention_probs_dropout_prob)
    super(Attention, self).build(unused_input_shapes)

  def reshape_to_matrix(self, input_tensor):
    ndims = input_tensor.shape.ndims
    if ndims < 2:
      raise ValueError("Input tensor must have at least rank 2."
                       "Shape = %s" % (input_tensor.shape))
    if ndims == 2:
      return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor

  def __call__(self, from_tensor, to_tensor, attention_mask=None, **kwargs):
    inputs = tf_utils.pack_inputs([from_tensor, to_tensor, attention_mask])
    return super(Attention, self).__call__(inputs, **kwargs)

  def call(self, inputs,**kwargs):
    (from_tensor, to_tensor, attention_mask) = tf_utils.unpack_inputs(inputs)
    query_tensor = self.query_dense(from_tensor)

    key_tensor = self.key_dense(to_tensor)

    value_tensor = self.value_dense(to_tensor)

    attention_scores = tf.einsum("BTNH,BFNH->BNFT", key_tensor, query_tensor)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(self.size_per_head)))

    if attention_mask is not None:
      attention_mask = tf.expand_dims(attention_mask, axis=[1])
      adder = (1.0 - tf.cast(attention_mask, attention_scores.dtype)) * -10000.0

      attention_scores += adder

    attention_probs = tf.nn.softmax(attention_scores)

    attention_probs = self.attention_probs_dropout(attention_probs,training=kwargs.get('training', False))

    context_tensor = tf.einsum("BNFT,BTNH->BFNH", attention_probs, value_tensor)

    return context_tensor

  def _projection_dense_layer(self, name):
    """A helper to define a projection layer."""
    return Dense3D(
        num_attention_heads=self.num_attention_heads,
        size_per_head=self.size_per_head,
        kernel_initializer=get_initializer(self.initializer_range),
        output_projection=False,
        backward_compatible=self.backward_compatible,
        name=name)


class Dense3D(tf.keras.layers.Layer):
  def __init__(self,
               num_attention_heads=12,
               size_per_head=72,
               kernel_initializer=None,
               bias_initializer="zeros",
               activation=None,
               use_bias=True,
               output_projection=False,
               backward_compatible=False,
               **kwargs):
    super(Dense3D, self).__init__(**kwargs)
    self.num_attention_heads = num_attention_heads
    self.size_per_head = size_per_head
    self.hidden_size = num_attention_heads * size_per_head
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.activation = activation
    self.use_bias = use_bias
    self.output_projection = output_projection
    self.backward_compatible = backward_compatible

  @property
  def compatible_kernel_shape(self):
    if self.output_projection:
      return [self.hidden_size, self.hidden_size]
    return [self.last_dim, self.hidden_size]

  @property
  def compatible_bias_shape(self):
    return [self.hidden_size]

  @property
  def kernel_shape(self):
    if self.output_projection:
      return [self.num_attention_heads, self.size_per_head, self.hidden_size]
    return [self.last_dim, self.num_attention_heads, self.size_per_head]

  @property
  def bias_shape(self):
    if self.output_projection:
      return [self.hidden_size]
    return [self.num_attention_heads, self.size_per_head]

  def build(self, input_shape):
    """Implements build() for the layer."""
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError("Unable to build `Dense3D` layer with non-floating "
                      "point (and non-complex) dtype %s" % (dtype,))
    input_shape = tf.TensorShape(input_shape)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError("The last dimension of the inputs to `Dense3D` "
                       "should be defined. Found `None`.")
    self.last_dim = tf.compat.dimension_value(input_shape[-1])
    self.input_spec = tf.keras.layers.InputSpec(
        min_ndim=3, axes={-1: self.last_dim})
    # Determines variable shapes.
    if self.backward_compatible:
      kernel_shape = self.compatible_kernel_shape
      bias_shape = self.compatible_bias_shape
    else:
      kernel_shape = self.kernel_shape
      bias_shape = self.bias_shape

    self.kernel = self.add_weight(
        "kernel",
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        dtype=self.dtype,
        trainable=True)
    if self.use_bias:
      self.bias = self.add_weight(
          "bias",
          shape=bias_shape,
          initializer=self.bias_initializer,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    super(Dense3D, self).build(input_shape)

  def call(self, inputs):
    if self.backward_compatible:
      kernel = tf.keras.backend.reshape(self.kernel, self.kernel_shape)
      bias = (tf.keras.backend.reshape(self.bias, self.bias_shape)
              if self.use_bias else None)
    else:
      kernel = self.kernel
      bias = self.bias

    if self.output_projection:
      ret = tf.einsum("abcd,cde->abe", inputs, kernel)
    else:
      ret = tf.einsum("abc,cde->abde", inputs, kernel)
    if self.use_bias:
      ret += bias
    if self.activation is not None:
      return self.activation(ret)
    return ret


class Dense2DProjection(tf.keras.layers.Layer):

  def __init__(self,
               output_size,
               kernel_initializer=None,
               bias_initializer="zeros",
               activation=None,
               fp32_activation=False,
               **kwargs):
    super(Dense2DProjection, self).__init__(**kwargs)
    self.output_size = output_size
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.activation = activation
    self.fp32_activation = fp32_activation

  def build(self, input_shape):
    """Implements build() for the layer."""
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError("Unable to build `Dense2DProjection` layer with "
                      "non-floating point (and non-complex) "
                      "dtype %s" % (dtype,))
    input_shape = tf.TensorShape(input_shape)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError("The last dimension of the inputs to "
                       "`Dense2DProjection` should be defined. "
                       "Found `None`.")
    last_dim = tf.compat.dimension_value(input_shape[-1])
    self.input_spec = tf.keras.layers.InputSpec(min_ndim=3, axes={-1: last_dim})
    self.kernel = self.add_weight(
        "kernel",
        shape=[last_dim, self.output_size],
        initializer=self.kernel_initializer,
        dtype=self.dtype,
        trainable=True)
    self.bias = self.add_weight(
        "bias",
        shape=[self.output_size],
        initializer=self.bias_initializer,
        dtype=self.dtype,
        trainable=True)
    super(Dense2DProjection, self).build(input_shape)

  def call(self, inputs):
    ret = tf.einsum("abc,cd->abd", inputs, self.kernel)
    ret += self.bias
    if self.activation is not None:
      if self.dtype == tf.float16 and self.fp32_activation:
        ret = tf.cast(ret, tf.float32)
      return self.activation(ret)
    return ret


class TransformerBlock(tf.keras.layers.Layer):
  def __init__(self,
               hidden_size=768,
               num_attention_heads=12,
               intermediate_size=3072,
               intermediate_activation="gelu",
               hidden_dropout_prob=0.0,
               attention_probs_dropout_prob=0.0,
               initializer_range=0.02,
               backward_compatible=False,
               float_type=tf.float32,
               **kwargs):
    super(TransformerBlock, self).__init__(**kwargs)
    self.hidden_size = hidden_size
    self.num_attention_heads = num_attention_heads
    self.intermediate_size = intermediate_size
    self.intermediate_activation = tf_utils.get_activation(
        intermediate_activation)
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_range = initializer_range
    self.backward_compatible = backward_compatible
    self.float_type = float_type

    if self.hidden_size % self.num_attention_heads != 0:
      raise ValueError(
          "The hidden size (%d) is not a multiple of the number of attention "
          "heads (%d)" % (self.hidden_size, self.num_attention_heads))
    self.attention_head_size = int(self.hidden_size / self.num_attention_heads)

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.attention_layer = Attention(
        num_attention_heads=self.num_attention_heads,
        size_per_head=self.attention_head_size,
        attention_probs_dropout_prob=self.attention_probs_dropout_prob,
        initializer_range=self.initializer_range,
        backward_compatible=self.backward_compatible,
        name="self_attention")
    self.attention_output_dense = Dense3D(
        num_attention_heads=self.num_attention_heads,
        size_per_head=int(self.hidden_size / self.num_attention_heads),
        kernel_initializer=get_initializer(self.initializer_range),
        output_projection=True,
        backward_compatible=self.backward_compatible,
        name="self_attention_output")
    self.attention_dropout = tf.keras.layers.Dropout(
        rate=self.hidden_dropout_prob)
    self.attention_layer_norm = (
        tf.keras.layers.LayerNormalization(
            name="self_attention_layer_norm", axis=-1, epsilon=1e-12,
            # We do layer norm in float32 for numeric stability.
            dtype=tf.float32))
    self.intermediate_dense = Dense2DProjection(
        output_size=self.intermediate_size,
        kernel_initializer=get_initializer(self.initializer_range),
        activation=self.intermediate_activation,
        # Uses float32 so that gelu activation is done in float32.
        fp32_activation=True,
        name="intermediate")
    self.output_dense = Dense2DProjection(
        output_size=self.hidden_size,
        kernel_initializer=get_initializer(self.initializer_range),
        name="output")
    self.output_dropout = tf.keras.layers.Dropout(rate=self.hidden_dropout_prob)
    self.output_layer_norm = tf.keras.layers.LayerNormalization(
        name="output_layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32)
    super(TransformerBlock, self).build(unused_input_shapes)

  def common_layers(self):
    """Explicitly gets all layer objects inside a Transformer encoder block."""
    return [
        self.attention_layer, self.attention_output_dense,
        self.attention_dropout, self.attention_layer_norm,
        self.intermediate_dense, self.output_dense, self.output_dropout,
        self.output_layer_norm
    ]

  def __call__(self, input_tensor, attention_mask=None, **kwargs):
    inputs = tf_utils.pack_inputs([input_tensor, attention_mask])
    return super(TransformerBlock, self).__call__(inputs, **kwargs)

  def call(self, inputs, **kwargs):
    """Implements call() for the layer."""
    (input_tensor, attention_mask) = tf_utils.unpack_inputs(inputs)
    attention_output = self.attention_layer(
        from_tensor=input_tensor,
        to_tensor=input_tensor,
        attention_mask=attention_mask,**kwargs)
    attention_output = self.attention_output_dense(attention_output)
    attention_output = self.attention_dropout(attention_output,training=kwargs.get('training', False))
    # Use float32 in keras layer norm and the gelu activation in the
    # intermediate dense layer for numeric stability
    attention_output = self.attention_layer_norm(input_tensor +
                                                 attention_output)
    if self.float_type == tf.float16:
      attention_output = tf.cast(attention_output, tf.float16)
    intermediate_output = self.intermediate_dense(attention_output)
    if self.float_type == tf.float16:
      intermediate_output = tf.cast(intermediate_output, tf.float16)
    layer_output = self.output_dense(intermediate_output)
    layer_output = self.output_dropout(layer_output,training=kwargs.get('training', False))
    # Use float32 in keras layer norm for numeric stability
    layer_output = self.output_layer_norm(layer_output + attention_output)
    if self.float_type == tf.float16:
      layer_output = tf.cast(layer_output, tf.float16)
    return layer_output


class Transformer(tf.keras.layers.Layer):

  def __init__(self,
               num_hidden_layers=12,
               hidden_size=768,
               num_attention_heads=12,
               intermediate_size=3072,
               intermediate_activation="gelu",
               hidden_dropout_prob=0.0,
               attention_probs_dropout_prob=0.0,
               initializer_range=0.02,
               backward_compatible=False,
               float_type=tf.float32,
               **kwargs):
    super(Transformer, self).__init__(**kwargs)
    self.num_hidden_layers = num_hidden_layers
    self.hidden_size = hidden_size
    self.num_attention_heads = num_attention_heads
    self.intermediate_size = intermediate_size
    self.intermediate_activation = tf_utils.get_activation(
        intermediate_activation)
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_range = initializer_range
    self.backward_compatible = backward_compatible
    self.float_type = float_type

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.layers = []
    for i in range(self.num_hidden_layers):
      self.layers.append(
          TransformerBlock(
              hidden_size=self.hidden_size,
              num_attention_heads=self.num_attention_heads,
              intermediate_size=self.intermediate_size,
              intermediate_activation=self.intermediate_activation,
              hidden_dropout_prob=self.hidden_dropout_prob,
              attention_probs_dropout_prob=self.attention_probs_dropout_prob,
              initializer_range=self.initializer_range,
              backward_compatible=self.backward_compatible,
              float_type=self.float_type,
              name=("layer_%d" % i)))
    super(Transformer, self).build(unused_input_shapes)

  def __call__(self, input_tensor, attention_mask=None, **kwargs):
    inputs = tf_utils.pack_inputs([input_tensor, attention_mask])
    return super(Transformer, self).__call__(inputs=inputs, **kwargs)

  def call(self, inputs, return_all_layers=False, **kwargs):
    unpacked_inputs = tf_utils.unpack_inputs(inputs)
    input_tensor = unpacked_inputs[0]
    attention_mask = unpacked_inputs[1]
    output_tensor = input_tensor

    all_layer_outputs = []
    for layer in self.layers:
      output_tensor = layer(output_tensor, attention_mask,**kwargs)
      all_layer_outputs.append(output_tensor)

    if return_all_layers:
      return all_layer_outputs

    return all_layer_outputs[-1]


def get_initializer(initializer_range=0.02):
  return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def create_attention_mask_from_input_mask(from_tensor, to_mask):
  from_shape = tf_utils.get_shape_list(from_tensor, expected_rank=[2, 3])
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  to_shape = tf_utils.get_shape_list(to_mask, expected_rank=2)
  to_seq_length = to_shape[1]

  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]),
      dtype=from_tensor.dtype)

  broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=from_tensor.dtype)

  mask = broadcast_ones * to_mask

  return mask
