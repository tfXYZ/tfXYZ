# Import from standard libraries
from __future__ import division, absolute_import, print_function
import tensorflow as tf, importlib
from six import iteritems

# Import from our libraries
import core.losses
from core.common import Channel, get_debug_session
from core.evaluation import basic_stats_numpy, classification_accuracy
from core.models import common_branch
from apps.base import BaseApp

# The flags
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('loss_function', 'core.losses.ce_loss', '')
tf.app.flags.DEFINE_string('numpy_channels', '', '')


class Classification(BaseApp):
  """
  Basic application class for a classification task with a single/multiple label(s), ce loss, an endpoint
  called 'bottleneck'
  """
  def __init__(self, n_classes, labels, name, dataset_dir, **kwargs):
    super(Classification, self).__init__(name, dataset_dir, **kwargs)
    self.test_samples = 1
    self.test_mb = 100
    self.raw_name = self.name.split('_')[0]

    # Just to allow non-lists in case of only one label
    if not isinstance(labels, list):
      self.labels = [labels]
      self.n_classes = [n_classes]
    else:
      self.labels = labels
      self.n_classes = n_classes

  def create_filename_queue(self, is_train, files, lengths):
    return tf.train.string_input_producer(files, capacity=500000)
  
  def precache_processing(self, is_train, val_dict):
    if is_train:
      return val_dict

    return {k: tf.expand_dims(v, 0) for k, v in iteritems(val_dict)}
  
  def postcache_processing(self, is_train, val_dict, tfr_structure):
    return val_dict

  def compute_loss(self, global_endpoints, module_endpoints):
    # Compute the different losses
    losses = []
    for l, n in zip(self.labels, self.n_classes):
      logit_name = '{}_{}_logits'.format(self.raw_name, l)
      
      sep_index = FLAGS.loss_function.rindex('.')
      containing_module = importlib.import_module(FLAGS.loss_function[:sep_index])
      loss_func = getattr(containing_module, FLAGS.loss_function[sep_index+1:])
      loss = loss_func(module_endpoints[logit_name], 
                       module_endpoints[l],
                       n_classes=n)
      losses.append(loss)

      # Add summaries
      loss_name = self.raw_name if len(self.labels) == 1 else '{}_{}'.format(self.raw_name, l)
      tf.summary.scalar('{}_loss'.format(loss_name), 
                        loss)

    return losses

  def monitoring_channels(self, is_train, global_endpoints, module_endpoints):
    channels = {}
    for l in self.labels:
      logit_name = '{}_{}_logits'.format(self.raw_name, l)
      logits = module_endpoints[logit_name]
      labels = module_endpoints[l]
      if is_train:
        accuracy_name = 'accuracy' if len(self.labels) == 1 else '{}_{}_accuracy'.format(self.raw_name, l)
        channels[accuracy_name] = classification_accuracy(is_train, logits, labels)
      else:
        channels[logit_name] = Channel(logits, False, True)
        channels['{}_{}_labels'.format(self.raw_name, l)] = Channel(labels, False, True)
    return channels

  def numpy_channels(self, concate_aggregators, step):
    ret = {}
    if FLAGS.numpy_channels:
      sep_index = FLAGS.numpy_channels.rindex('.')
      containing_module = importlib.import_module(FLAGS.numpy_channels[:sep_index])
      additional_channels = getattr(containing_module, FLAGS.numpy_channels[sep_index+1:])
      ret = additional_channels(self, concate_aggregators, step)
      
    for l in self.labels:
      logit_name = '{}_{}_logits'.format(self.raw_name, l)
      logits = concate_aggregators[logit_name]
      labels = concate_aggregators['{}_{}_labels'.format(self.raw_name, l)]
      probs, preds, labels, acc, uar, conf_mat, auprc, acc_at_3, acc_at_5 = basic_stats_numpy(logits, labels, self.test_samples) 
      name_prefix = '' if len(self.labels) == 1 else '{}_'.format(l)
      ret.update({'{}UAR'.format(name_prefix): uar, 
                  '{}acc'.format(name_prefix): acc, 
                  '{}conf_mat'.format(name_prefix): conf_mat, 
                  '{}auprc'.format(name_prefix): auprc, 
                  '{}acc@3'.format(name_prefix): acc_at_3, 
                  '{}acc@5'.format(name_prefix): acc_at_5})
    return ret

  def top_layers(self, is_train, global_endpoints, module_endpoints):
    inputs = module_endpoints['bottleneck']

    # The top layers
    for l, n in zip(self.labels, self.n_classes):
      out = common_branch(is_train, inputs, n, '{}_{}_top'.format(self.raw_name, l))
      logit_name = '{}_{}_logits'.format(self.raw_name, l)
      module_endpoints[logit_name] = out
