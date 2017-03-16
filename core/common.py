from __future__ import division, absolute_import, print_function
import tensorflow as tf, arff, numpy as np, importlib
import codecs
from scipy import stats
FLAGS = tf.app.flags.FLAGS

# Some constants
BN_COLLECTION = 'bn_updates'
SUMMARIES_COLLECTION = 'to_summarize'
MOVING_AVG_COLLECTION = 'moving_avg'
NON_TRAIN_VARS = 'non_train_vars'
NON_INIT_VARS = 'non_init_vars'

def bayes_classifier_with_discretization(features, labels, eval_features, bins=100, balanced_classes=False):
  n_classes = labels.max()+1
  n_features = features.shape[1]
  n_eval_examples = len(eval_features)
  
  # Find p(C_k)
  label_counts = np.bincount(labels)
  if not balanced_classes:
    p_c = label_counts / float(label_counts.sum())
  else:
    p_c = np.ones_like(label_counts) / float(len(label_counts))
  
  # Find the bins for each feature
  all_bins = []
  test_bin_ids = []
  eval_probs = np.zeros((n_eval_examples, n_classes)) # (ex x class)
  for i in range(0, n_features):
    x_i = features[:, i]
    # The bins used for x_i, computed with all classes
    x_i_bins = stats.mstats.mquantiles(x_i, [t/float(bins) for t in range(0, bins+1)])
    x_i_bins = list(set(x_i_bins))
    x_i_bins.sort()
    x_i_bins = np.array(x_i_bins)
    x_i_bins[0] = -np.Infinity; x_i_bins[-1] = np.Infinity
    all_bins.append(x_i_bins)
    # The bin IDs for the test data
    # since the first bin is -infty, digitize returns >=1, and we subtract 1 to get the bin number
    eval_x_i_discrete = np.digitize(eval_features[:, i], x_i_bins) - 1
    test_bin_ids.append(eval_x_i_discrete)
  
  # Find p(x_1, ... , x_n | C_j)  
  for j in range(0, n_classes): 
    x_c_j = features[labels == j]
    # The counts of x_i and c_j
    x_c_j_counts, _ = np.histogramdd(x_c_j, bins=all_bins)
    if label_counts[j] != 0:
      x_c_j_probs = x_c_j_counts / float(label_counts[j]) # Only inside the bins
    else:
      x_c_j_probs = x_c_j_counts # All zeros
    eval_probs[:, j] = x_c_j_probs[test_bin_ids]
      
  # Multiply everything
  out = np.tile(p_c.reshape([1, len(p_c)]), [n_eval_examples, 1])
  out = out * eval_probs
    
  # Normalize
  out = out / np.sum(out, axis=1).reshape([len(eval_features), 1])
  out = np.where(np.isnan(out), 1 / float(len(label_counts)), out)
  
  return out

def naive_bayes_classifier_with_discretization(features, labels, eval_features, bins=100, balanced_classes=False, smooth=0):
  # Find p(C_k)
  label_counts = np.bincount(labels)
  if not balanced_classes:
    p_c = label_counts / float(label_counts.sum())
  else:
    p_c = np.ones_like(label_counts) / float(len(label_counts))
  
  # Find p(x_i | C_j)
  eval_probs = np.zeros((features.shape[1], labels.max()+1, len(eval_features))) # (feature x class x ex)
  for i in range(0, features.shape[1]):
    x_i = features[:, i]
    # The bins used for x_i, computed with all classes
    x_i_bins = stats.mstats.mquantiles(x_i, [t/float(bins) for t in range(0, bins+1)])
    x_i_bins = list(set(x_i_bins))
    x_i_bins.sort()
    x_i_bins = np.array(x_i_bins)
    x_i_bins[0] = -np.Infinity; x_i_bins[-1] = np.Infinity
    eval_x_i_discrete = np.digitize(eval_features[:, i], x_i_bins) - 1 # since the first bin is -infty, digitize returns >=1, and we subtract 1 to get the bin number
    for j in range(0, labels.max()+1):
      x_i_c_j = x_i[labels == j]
      # The counts of x_i and c_j
      x_i_c_j_counts, _ = np.histogram(x_i_c_j, x_i_bins)
      if label_counts[j] != 0:
        x_i_c_j_probs = x_i_c_j_counts / float(label_counts[j]) # Only inside the bins
        if smooth > 0:
          padded = np.hstack([[x_i_c_j_probs[0]]*smooth, x_i_c_j_probs, [x_i_c_j_probs[-1]]*smooth])
          x_i_c_j_probs = np.convolve(padded, np.ones([smooth*2+1]) / float(smooth*2+1), mode='valid')
      else:
        x_i_c_j_probs = x_i_c_j_counts # All zeros
      eval_probs[i, j] = x_i_c_j_probs[eval_x_i_discrete]
      
  # Multiply everything
  out = np.tile(p_c.reshape([len(p_c), 1]), [1, len(eval_features)])
  for k in range(0, features.shape[1]):
    out = out * eval_probs[k, :, :]
    
  # Normalize
  out = np.transpose(out) # Now it's (ex x class)
  out = out / np.sum(out, axis=1).reshape([len(eval_features), 1])
  out = np.where(np.isnan(out), 1 / float(len(label_counts)), out)
  
  return out

class Channel():
  def __init__(self, tensor, printable, concatenate):
    self.tensor = tensor
    self.printable = printable
    self.concatenate = concatenate

def class_counts(arff_file):
  data = arff.load(codecs.open(arff_file, 'r', 'utf-8'))
  counts = {}
  atts = [x[0] for x in data['attributes'][1:]]
  for att in atts:
    counts[att] = {}
  
  for row in data['data']:
    for i, label in enumerate(row[1:]):
      if not label in counts[atts[i]]:
        counts[atts[i]][label] = 1
      else: 
        counts[atts[i]][label] += 1
  
  n_examples = sum([counts[atts[0]][k] for k in counts[atts[0]].keys()])

  return counts, n_examples

def make_summaries():
  with tf.name_scope('summaries'):
    for name, tensor in tf.get_collection(SUMMARIES_COLLECTION):
      if len(tensor.get_shape()) == 0:
        tf.summary.scalar(name, tensor)
      else:
        mean = tf.reduce_mean(tensor)
        std = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
        tf.summary.scalar(name + '/mean', mean)
        tf.summary.scalar(name + '/std', std)
        tf.summary.histogram(name + '/activations', tensor)
        tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(tensor))


def load_apps_from_string(s):
  tf.logging.set_verbosity(tf.logging.ERROR)
  
  # Load the different apps
  apps = []
  for name in s.split(','):
    if name:
      # Get the app name
      if '[' in name:
        module_and_class_name = name[:name.index('[')]
      else:
        module_and_class_name = name
      
      # Parse module parameters
      attrs_parsed = []
      if '[' in name:
        attrs = name[name.index('[')+1:-1].split(':')
        for attr in attrs:
          attr_name = attr.split('=')[0]
          attr_val_string = attr.split('=')[1]
          
          if attr_val_string == 'True' or attr_val_string == 'False':
            attr_val = True if attr_val_string == 'True' else False
          else:
            try:
              attr_val = int(attr_val_string)
            except ValueError:
              try:
                attr_val = float(attr_val_string)
              except ValueError:
                attr_val = attr_val_string
          attrs_parsed.append((attr_name, attr_val))
          
      # Set the train_files and eval_files parameters:
      # Priotities are: 1) explicitly from the apps flag, 2) train/eval_files flag 3) default from class
      has_train_files = False
      has_eval_files = False
      for attr in attrs_parsed:
        if attr[0] == 'train_files':
          has_train_files = True
        if attr[0] == 'eval_files':
          has_eval_files = True
      if not has_train_files and FLAGS.train_files:
        attrs_parsed.append(('train_files', FLAGS.train_files))
      if not has_eval_files and FLAGS.eval_files:
        attrs_parsed.append(('eval_files', FLAGS.eval_files))  
  
      # Create module or class instance
      kwargs = {attr[0]:attr[1] for attr in attrs_parsed}
      module_name, class_name = module_and_class_name.split('.')
      containing_module = importlib.import_module('apps.{}'.format(module_name))
      class_name = getattr(containing_module, class_name)
      app = class_name(**kwargs)
  
      # Add module to list
      apps.append({'app': app})
    
  return apps


def get_debug_session():
  s = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  s.run(tf.variables_initializer(list(set(tf.global_variables()) - set(tf.get_collection(NON_INIT_VARS)))))
  coord = tf.train.Coordinator()
  tf.train.start_queue_runners(sess=s, coord=coord)   
  return s