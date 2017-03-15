from __future__ import absolute_import, division, print_function

import tensorflow as tf

from .common import get_debug_session
from .blocks import *

# The flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('kernel_size', 5, '')
def model_multitask1(is_train, sess, module_endpoints):
  global_endpoints = {}
  inputs = tf.concat(axis=0, values=[e['bottleneck_inputs'] for e in module_endpoints])
  global_endpoints['inputs'] = inputs
  out = inputs
  
  n_features = inputs.get_shape()[2].value
  w_std = FLAGS.w_std
  b_init = FLAGS.b_init
  
  # The layers
  if not FLAGS.preprocess_layers:
    out = conv2d(out, [FLAGS.kernel_size, n_features], [1,1,1,1], FLAGS.n_fmaps, w_std, b_init, True, 'VALID', 'conv1')
    global_endpoints['pre'] = out
    out = batch_norm(is_train, out, 'conv1') if FLAGS.bn else out
    global_endpoints['post'] = out
    out = tf.nn.relu(out)

  out = conv2d(out, [1, 1], [1,1,1,1], FLAGS.n_fmaps, w_std, b_init, True, 'VALID', 'conv1_1')
  out = batch_norm(is_train, out, 'conv1_1') if FLAGS.bn else out
  out = tf.nn.relu(out)
  global_endpoints['conv1_1'] = out
  
  out = tf.nn.max_pool(out, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME', name='pool1')
  
  out = conv2d(out, [FLAGS.kernel_size, 1], [1,1,1,1], FLAGS.n_fmaps, w_std, b_init, True, 'VALID', 'conv2')
  out = batch_norm(is_train, out, 'conv2') if FLAGS.bn else out
  out = tf.nn.relu(out)
  
  out = conv2d(out, [1, 1], [1,1,1,1], FLAGS.n_fmaps, w_std, b_init, True, 'VALID', 'conv2_1')
  out = batch_norm(is_train, out, 'conv2_1') if FLAGS.bn else out
  out = tf.nn.relu(out)
  global_endpoints['conv2_1'] = out
    
  out = tf.nn.max_pool(out, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME', name='pool2')
  
  out = conv2d(out, [FLAGS.kernel_size, 1], [1,1,1,1], FLAGS.n_fmaps, w_std, b_init, True, 'VALID', 'conv3')
  out = batch_norm(is_train, out, 'conv3') if FLAGS.bn else out
  out = tf.nn.relu(out)
  
  out = conv2d(out, [1, 1], [1,1,1,1], FLAGS.n_fmaps, w_std, b_init, True, 'VALID', 'conv3_1')
  out = batch_norm(is_train, out, 'conv3_1') if FLAGS.bn else out
  out = tf.nn.relu(out)
  global_endpoints['conv3_1'] = out
    
  out = tf.nn.max_pool(out, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME', name='pool3')
  
  out = dense(flatten(out), FLAGS.n_dense_units, w_std, b_init, True, 'dense1')
  out = batch_norm(is_train, out, 'dense1') if FLAGS.bn else out
  out = tf.nn.relu(out)
  global_endpoints['dense1'] = out
  out = tf.nn.dropout(out, 0.5) if is_train and FLAGS.dropout else out
    
  out = dense(flatten(out), FLAGS.n_dense_units, w_std, b_init, True, 'dense2')
  out = batch_norm(is_train, out, 'dense2') if FLAGS.bn else out
  out = tf.nn.relu(out)
  global_endpoints['dense2'] = out
  out = tf.nn.dropout(out, 0.5) if is_train and FLAGS.dropout else out
  
  # Set the endpoints names
  global_endpoints['bottleneck'] = out

  return global_endpoints


def model_one_recurrent(is_train, sess, module_endpoints):
  global_endpoints = {}
  inputs = tf.concat(axis=0, values=[e['bottleneck_inputs'] for e in module_endpoints])
  global_endpoints['inputs'] = inputs
  
  w_std = FLAGS.w_std
  b_init = FLAGS.b_init
  
  with tf.variable_scope('rec_cell', initializer=tf.random_normal_initializer(0.0, w_std)):
    cell = tf.contrib.rnn.GRUCell(num_units=FLAGS.recurrent_dim)
    inputs_as_list = tf.unstack(tf.transpose(tf.squeeze(inputs), [1,0,2]))
    outputs, state = tf.contrib.rnn.static_rnn(cell, inputs_as_list, dtype=tf.float32)
  
  # Set the endpoints names
  global_endpoints['bottleneck'] = outputs[-1]

  return global_endpoints


def just_recurrent(is_train, sess, module_endpoints):
  global_endpoints = {}
  inputs = tf.concat(axis=0, values=[e['bottleneck_inputs'] for e in module_endpoints])
  global_endpoints['inputs'] = inputs
  
  w_std = FLAGS.w_std
  b_init = FLAGS.b_init
  
  with tf.variable_scope('rec_cell', initializer=tf.random_normal_initializer(0.0, w_std)):
    cell = tf.contrib.rnn.GRUCell(num_units=FLAGS.recurrent_dim)
    if FLAGS.n_recurrent > 1:
      cell = tf.contrib.rnn.MultiRNNCell([cell] * FLAGS.n_recurrent)
  
  inputs_as_list = tf.unstack(tf.transpose(tf.squeeze(inputs), [1,0,2]))
  outputs, state = tf.contrib.rnn.static_rnn(cell, inputs_as_list, dtype=tf.float32)
  
  # Set the endpoints names
  global_endpoints['bottleneck'] = outputs[-1]

  return global_endpoints


def model_simple(is_train, sess, module_endpoints):
  global_endpoints = {}
  inputs = tf.concat(axis=0, values=[e['bottleneck_inputs'] for e in module_endpoints])
  global_endpoints['inputs'] = inputs
  
  w_std = FLAGS.w_std
  b_init = FLAGS.b_init
  
  out = inputs
  out = flatten(out)
  out = dense(out, 300, w_std, b_init, True, 'dense')
  out = tf.nn.relu(out)
  
  # Set the endpoints names
  global_endpoints['bottleneck'] = out

  return global_endpoints


def model_just_dense(is_train, sess, module_endpoints):
  global_endpoints = {}
  inputs = tf.concat(axis=0, values=[e['features'] for e in module_endpoints])
  global_endpoints['inputs'] = inputs
  
  w_std = FLAGS.w_std
  b_init = FLAGS.b_init
  dr = FLAGS.dropout and is_train
  
  out = inputs
  out = flatten(out)
  for i in range(0, FLAGS.n_dense):
    out = dense(out, FLAGS.n_dense_units, w_std, b_init, not FLAGS.bn, 'dense{}'.format(i))
    out = batch_norm(is_train, out, 'dense{}'.format(i)) if FLAGS.bn else out
    out = tf.nn.relu(out)
    tf.nn.dropout(out, 0.5) if dr else out
  
  # Set the endpoints names
  global_endpoints['bottleneck'] = out

  return global_endpoints


def common_branch(is_train, inputs, n_classes, scope):
  out = inputs
  for i in range(0, FLAGS.branch_layers):
    scope_name = scope + '_dense{}'.format(i)
    out = dense(out, FLAGS.branch_dim, FLAGS.w_std, FLAGS.b_init, True, scope_name)
    out = batch_norm(is_train, out, scope_name)
  out = dense(out, n_classes, 0.0, 0.0, True, scope + '_softmax')
  return out


def common_first_layers(is_train, module_endpoints, scope):
  if FLAGS.preprocess_layers:
    inputs = module_endpoints['features']
    n_features = inputs.get_shape()[2].value    
    scope_prefix = scope

    out = inputs
    out = conv2d(out, [5,n_features], [1,1,1,1], FLAGS.n_fmaps, FLAGS.w_std, FLAGS.b_init, True, 'VALID', scope_prefix + '_conv1')
    out = batch_norm(is_train, out, scope_prefix + '_conv1')
    out = tf.nn.relu(out)
    
    module_endpoints['bottleneck_inputs'] = out
  else:
    if 'features' in module_endpoints.keys():
      module_endpoints['bottleneck_inputs'] = module_endpoints['features']
