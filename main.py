# Import from standard libraries
from __future__ import division, absolute_import, print_function
import tensorflow as tf

# Import from our libraries
from core.common import get_debug_session, load_apps_from_string
from core import models
from core.models import *
from core.train import main_loop
from apps import *
from local import *

# The flags
FLAGS = tf.app.flags.FLAGS

# Environment settings
tf.app.flags.DEFINE_string('summary_dir', '', 'Path to directory where the training summaries should be stored.')
tf.app.flags.DEFINE_string('checkpoint_dir', '', 'Path to directory where the model checkpoints should be stored.')
tf.app.flags.DEFINE_string('model_name', '', 'Name of the checkpoints files.')
tf.app.flags.DEFINE_string('gpu', '0', 'ID of gpu to use.')
tf.app.flags.DEFINE_boolean('log_device_placement', False, """True or False.""")

# Hyperparams and other constants
tf.app.flags.DEFINE_float('lr', 0.1, """Learning rate.""")
tf.app.flags.DEFINE_float('mom', 0.0, """Momentum.""")
tf.app.flags.DEFINE_integer('batches', 300000, """Number of training batches before stopping.""")
tf.app.flags.DEFINE_string('alg', 'sgd', """Training algorithm to use.""")
tf.app.flags.DEFINE_integer('decay_steps', 0, '')
tf.app.flags.DEFINE_float('decay_rate', 0.0, '')
tf.app.flags.DEFINE_boolean('bn', True, 'Batch normalization.')
tf.app.flags.DEFINE_float('gc', 10.0, """Gradient Clipping.""")
tf.app.flags.DEFINE_float('w_std', 0.01, 'Standard deviation of weight initialization.')
tf.app.flags.DEFINE_float('b_init', 0.0, 'Value for bias initialization.')
tf.app.flags.DEFINE_integer('branch_layers', 0, '')
tf.app.flags.DEFINE_integer('branch_dim', 500, '')
tf.app.flags.DEFINE_boolean('finetune_all', True, '')
tf.app.flags.DEFINE_boolean('restore_all', True, '')
tf.app.flags.DEFINE_boolean('preprocess_layers', False, '')
tf.app.flags.DEFINE_integer('train_mb', 64, '')
tf.app.flags.DEFINE_float('bn_decay', 0.95, '')
tf.app.flags.DEFINE_boolean('dropout', False, '')
tf.app.flags.DEFINE_integer('simple_layers', 3, '')
tf.app.flags.DEFINE_integer('simple_dim', 500, '')
tf.app.flags.DEFINE_float('w_sim_loss_factor', -1, '')
tf.app.flags.DEFINE_string('domain_adv_tensor', '', '')
tf.app.flags.DEFINE_float('domain_loss_factor', 0.0, '')
tf.app.flags.DEFINE_string('act_sim_tensors', '', '')
tf.app.flags.DEFINE_string('act_sim_loss_factors', '0.0,0.0,0.0', '')
tf.app.flags.DEFINE_boolean('monitor_grad_norms', False, '')
tf.app.flags.DEFINE_boolean('eval_before_training', False, '')
tf.app.flags.DEFINE_boolean('dump_activations', False, '')
tf.app.flags.DEFINE_integer('crop_strides', 1, '')
tf.app.flags.DEFINE_integer('n_recurrent', 3, '')
tf.app.flags.DEFINE_integer('recurrent_dim', 512, '')
tf.app.flags.DEFINE_integer('n_fmaps', 256, '')
tf.app.flags.DEFINE_integer('n_conv', 2, '')
tf.app.flags.DEFINE_integer('n_nin_units', 256, '')
tf.app.flags.DEFINE_integer('n_dense', 2, '')
tf.app.flags.DEFINE_integer('n_dense_units', 500, '')
tf.app.flags.DEFINE_integer('save_every', 1500, """save a checkpoint every how many batches.""")
tf.app.flags.DEFINE_integer('eval_every', 1500, """monitor on the test set every how many batches.""")
tf.app.flags.DEFINE_integer('train_monitor_every', 250, """monitor on the train set every how many batches.""")
tf.app.flags.DEFINE_integer('train_summary_every', 250, """summarize on the train set every how many batches.""")
tf.app.flags.DEFINE_boolean('cache', True, '')

# Path to restore weights for the bottleneck part from
tf.app.flags.DEFINE_string('restore_path', '', '') # baseline cifar10

tf.app.flags.DEFINE_string('train_files', '', 'Name of file that contains the list of training files.')
tf.app.flags.DEFINE_string('eval_files', '', 'Name of file that contains the list of evaluation files.')
tf.app.flags.DEFINE_string('apps', '', 'Comma-separated list of python classes that will be executed.')
tf.app.flags.DEFINE_string('eval_apps', '', '')
tf.app.flags.DEFINE_string('train_apps', '', '')

# The bottleneck inference function
tf.app.flags.DEFINE_string('inference_bottleneck', 'model_cifar', 'Function that defines the network structure.')


def main(action_manager=None):
  train_and_eval_apps = load_apps_from_string(FLAGS.apps)
  eval_apps = load_apps_from_string(FLAGS.eval_apps)
  train_apps = load_apps_from_string(FLAGS.train_apps)
  
  # Override default parameters with specific apps values
  for m in train_and_eval_apps + eval_apps + train_apps:
    m['train_mb'] = m['app'].train_mb if hasattr(m['app'], 'train_mb') else FLAGS.train_mb
    m['cache'] = m['app'].cache if hasattr(m['app'], 'cache') else FLAGS.cache 
  
  # Set the final apps lists
  train_apps = train_and_eval_apps + train_apps
  eval_apps = eval_apps + [{'app': x['app'], 'cache': x['cache']} for x in train_and_eval_apps]
  
  # Do Training
  best_test_channels, best_step = main_loop(train_apps, eval_apps, action_manager)
  return best_test_channels, best_step, FLAGS

if __name__ == '__main__':
  main()
