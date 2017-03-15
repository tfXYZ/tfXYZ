from __future__ import division, absolute_import, print_function
import tensorflow as tf

from .common import BN_COLLECTION, get_debug_session
from six.moves import reduce

FLAGS = tf.app.flags.FLAGS


def dense(tensor, out_dim, w_std, b_init, with_bias, scope_name):
  """
  Implement a (1D) dense layer.

  Args:
    tensor: Input for the dense layer.
    out_dim: Number of units in the layer.
    w_std: Standard deviation for the initial values of the weights.
    b_init: Initial value for the biases.
    with_bias: Whether or not to use a bias.
    scope_name: Name to disambiguate the variables.

  Returns
    Output of the dense layer.
  """
  in_dim = tensor.get_shape()[1].value
  
  with tf.variable_scope(scope_name):
    w = tf.get_variable('w', [in_dim, out_dim], initializer=tf.truncated_normal_initializer(stddev=w_std))
    out = tf.matmul(tensor, w)
    if with_bias:
      b = tf.get_variable('b', [1, out_dim], initializer=tf.constant_initializer(b_init))
      out += b
    return out


def conv2d(tensor, kernel_shape, strides, out_channels, w_std, b_init, with_bias, padding, scope_name):
  """
  Implement a 2d convolutional layer.

  Args:
    tensor: Input for the convolutional layer. Shape should be [batch, height, width, channels].
    kernel_shape: Size of the kernel. Shape should be [height, width].
    strides: Strides for the convolution. Shape should be [batch_stride, height_stride, weight_stride,
        channel_stride].
    out_channels: Number of feature maps to extract.
    w_std: Standard deviation for the initial values of the weights.
    b_init: Initial value for the biases.
    with_bias: Whether or not to use a bias.
    padding: 'VALID' for no padding, 'SAME' for padding.
    scope_name: Name to disambiguate the variables.

  Returns:
    Output of the convolution.
  """
  in_channels = tensor.get_shape()[3].value
  
  with tf.variable_scope(scope_name):
    kernel = tf.get_variable('w', kernel_shape + [in_channels, out_channels], initializer=tf.truncated_normal_initializer(stddev=w_std))
    out = tf.nn.conv2d(tensor, kernel, strides, padding=padding)
    if with_bias:
      b = tf.get_variable('b', [1,1,1,out_channels], initializer=tf.constant_initializer(b_init))
      out += b
    return out


def flatten(tensor):
  """
  Flatten a tensor into two dimensions: [batch, size].

  Args:
    tensor: The tensor to flatten. Should be at least 2-dimensional.

  Returns:
    The flattened tensor.
  """
  sh = tensor.get_shape()
  other_dimensions = 1
  for i in range(1, len(sh)):
    other_dimensions *= sh[i].value
  return tf.reshape(tensor, [-1, other_dimensions])


def batch_norm(is_train, tensor, scope_name):
  """
  Implement a batch normalization operation on the input tensor.

  Args:
    is_train: True when constructing the training graph.
    tensor: Tensor on which to apply batch normalization.
    scope_name: Name to disambiguate the variables.

  Returns:
    Normalized tensor.
  """
  with tf.variable_scope(scope_name):
    BATCHNORM_MOVING_AVERAGE_DECAY = FLAGS.bn_decay
    # Get the shape of mean, variance, beta, gamma
    mask_shape = [1] * len(tensor.get_shape())
    mask_shape[-1] = tensor.get_shape()[-1].value
    
    # Create trainable variables to hold beta and gamma 
    beta = tf.get_variable('beta', mask_shape, initializer=tf.constant_initializer(0.0))   
    gamma = tf.get_variable('gamma', mask_shape, initializer=tf.constant_initializer(1.0)) 
  
    # Create non-trainable variables for the population mean and variance, and add them to the 
    pop_mean = tf.get_variable('pop_mean', mask_shape, initializer=tf.constant_initializer(0.0), trainable=False)
    pop_variance = tf.get_variable('pop_variance', mask_shape, initializer=tf.constant_initializer(1.0),
                                   trainable=False)
    
    if is_train:
      # Calculate the moments based on the individual batch.
      n_dims = len(tensor.get_shape())
      mean, variance = tf.nn.moments(x=tensor, axes=[i for i in range(0, n_dims -1)], keep_dims=True)
      
      # Update the population mean and variance
      pop_mean_update = pop_mean.assign(
        (BATCHNORM_MOVING_AVERAGE_DECAY * pop_mean + (1-BATCHNORM_MOVING_AVERAGE_DECAY) * mean)).op
      pop_variance_update = pop_variance.assign(
        (BATCHNORM_MOVING_AVERAGE_DECAY * pop_variance + (1-BATCHNORM_MOVING_AVERAGE_DECAY) * variance)).op
      
      # Add the update ops to a collection, in order to perform them when training
      tf.add_to_collection(BN_COLLECTION, pop_mean_update)
      tf.add_to_collection(BN_COLLECTION, pop_variance_update)
    else:
      # Just use the moving_mean and moving_variance.
      mean = pop_mean
      variance = pop_variance
      
    # Normalize the activations
    outputs = tf.nn.batch_normalization(tensor, mean, variance, beta, gamma, variance_epsilon=0.001)
  
    return outputs
  
def reverse_gradient(tensor, gradient_factor):
  return -gradient_factor*tensor + tf.stop_gradient((gradient_factor+1)*tensor)
  
def gather_nd(params, indices, name=None):
  shape = params.get_shape().as_list()
  rank = len(shape)
  flat_params = tf.reshape(params, [-1])
  multipliers = [reduce(lambda x, y: x*y, shape[i+1:], 1) for i in range(0, rank)]
  indices_unpacked = tf.unstack(tf.transpose(indices, [rank - 1] + range(0, rank - 1), name))
  flat_indices = sum([a*b for a,b in zip(multipliers, indices_unpacked)])
  return tf.gather(flat_params, flat_indices, name=name)


def crnn(tensor, kernel_size, stride, out_channels, rnn_n_layers, rnn_type, bidirectional, w_std, padding, scope_name):
  """
  Performs the 1-D Convolutional RNN Operation, according to the paper:
  Convolutional RNN: an Enhanced Model for Extracting Features from Sequential Data (https://arxiv.org/abs/1602.05875)
  Gil Keren and Bjoern Schuller.

  Calling the below function is equivalnet to applying one CRNN layer. For a deep model with a few
  CRNN layers, the function should be invoked multiple times.

  Given a tensor, the function extracts patches of `kernel_size` time-steps, and processed each
  with one or more recurrent layers. The hidden state of the recurrent neural network is then
  returned as the feature vector representing the path.

  Args:
    tensor: The tensor to perform the operation on, shape `[batch, time-steps, features]`
            or `[batch, time-steps, features, 1]`.
    kernel_size: The number of time-steps to include in every patch/window (same as in standard 1-D convolution).
    stride: the number of time-steps between two consecutive patches/windows (same as in standard 1-D convolution).
    out_channels: The number of extracted features from each patch/window (in standard 1-D convolution
                  known as the number of feature maps), which is the hidden dimension of the recurrent
                  layers that processes each patch/window.
    rnn_n_layers: The number of recurrent layers to process the patches/windows.
      (in the original paper was always =1).
    rnn_type: Type of recurrent layers to use: `simple`/`lstm`/`gru`
    bidirectional: Whether to use a bidirectional recurrent layers (such as BLSTM, when the rnn_type is 'lstm').
                   If True, The actual number of extracted features from each patch/window is `2 * out_channels`.
    w_std: Weights in the recurrent layers will be initialized randomly using a Gaussaian distribution with
           zero mean and a standard deviation of `w_std`. Biases are initialized with zero.
    padding: `SAME` or `VALID` (same as in standard 1-D convolution).
    scope_name: For variable naming, the name prefix for variables names.

  Returns:
    A 3-D `Tensor` with shape `[batch, time-steps, features]`, similarly to the output of a standard 1-D convolution.
  """
  with tf.variable_scope(scope_name, initializer=tf.truncated_normal_initializer(stddev=w_std)):
    # Expand to have 4 dimensions if needed
    if len(tensor.shape) == 3:
      tensor = tf.expand_dims(tensor, 3)
    
    # Extract the patches (returns [batch, time-steps, 1, patch content flattened])
    # batch_size = tensor.shape[0].value
    batch_size = tf.shape(tensor)[0]
    n_in_features = tensor.shape[2].value
    patches = tf.extract_image_patches(images=tensor, 
                             ksizes=[1, kernel_size, n_in_features, 1], 
                             strides=[1, stride, n_in_features, 1], 
                             rates=[1, 1, 1, 1], 
                             padding=padding)
    patches = patches[:, :, 0, :]
    
    # Reshape to do: 
    # 1) reshape the flattened patches back to [kernel_size, n_in_features]
    # 2) combine the batch and time-steps dimensions (which will be the new 'batch' size, for the RNN)
    # now shape will be [batch * time-steps, kernel_size, n_features]
    time_steps_after_stride = patches.shape[1].value
    patches = tf.reshape(patches, tf.stack([batch_size * time_steps_after_stride,
                                            kernel_size, n_in_features]))
    
    # Transpose and convert to a list, to fit the tf.contrib.rnn.static_rnn requirements
    # Now will be a list of length kernel_size, each element of shape [batch * time-steps, n_features]
    patches = tf.unstack(tf.transpose(patches, [1, 0, 2]))
    
    # Create the RNN Cell
    if rnn_type == 'simple':
      rnn_cell_func = tf.contrib.rnn.BasicRNNCell
    elif rnn_type == 'lstm':
      rnn_cell_func = tf.contrib.rnn.LSTMBlockCell
    elif rnn_type == 'gru':
      rnn_cell_func = tf.contrib.rnn.GRUBlockCell
    else:
      raise ValueError
    if not bidirectional:
      rnn_cell = rnn_cell_func(out_channels)
    else:
      rnn_cell_f = rnn_cell_func(out_channels)
      rnn_cell_b = rnn_cell_func(out_channels)
      
    # Multilayer RNN? (does not appear in the original paper)
    if rnn_n_layers > 1:
      if not bidirectional:
        rnn_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell] * rnn_n_layers)
      else:
        rnn_cell_f = tf.contrib.rnn.MultiRNNCell([rnn_cell_f] * rnn_n_layers)
        rnn_cell_b = tf.contrib.rnn.MultiRNNCell([rnn_cell_b] * rnn_n_layers)
    
    # The RNN itself
    if not bidirectional:
      outputs, state = tf.contrib.rnn.static_rnn(rnn_cell, patches, dtype=tf.float32)
    else:
      outputs, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(rnn_cell_f, rnn_cell_b, patches, dtype=tf.float32)
    
    # Use only the output of the last time-step (shape will be [batch * time-steps, out_channels]).
    # In the case of a bidirectional RNN, we want to take the last time-step of the forward RNN, 
    # and the first time-step of the backward RNN. 
    if not bidirectional:
      outputs = outputs[-1]
    else:
      half = int(outputs[0].shape.as_list()[-1] / 2)
      outputs = tf.concat([outputs[-1][:,:half], 
                           outputs[0][:,half:]], 
                          axis=1)
    
    # Expand the batch * time-steps back (shape will be [batch_size, time_steps, out_channels]
    if bidirectional:
      out_channels = 2 * out_channels
    outputs = tf.reshape(outputs, [batch_size, time_steps_after_stride, out_channels])
    
    return outputs
