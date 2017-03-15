"""
Numpy based evaluations
"""

import numpy as np, tensorflow as tf
from sklearn.metrics import precision_recall_curve, auc


def basic_stats_numpy(logits, labels, test_samples=1, op='mean'):
  # Probs
  probs = softmax_numpy(logits, temperature=1.0)
  if test_samples > 1:
    probs = average_or_max_tensor_numpy(probs, test_samples, op)
    labels = downsample_labels_numpy(labels, test_samples)
  # Preds
  preds = np.argmax(probs, axis=1)
  # conf_mat
  conf_mat = confusion_matrix_np(preds, labels, normalized=True)
  # acc
  acc = (preds == labels).mean()
  acc_at_3 = accuracy_at_k(probs, labels, 3)
  acc_at_5 = accuracy_at_k(probs, labels, 5)
  # uar
  uar = (conf_mat.diagonal() / conf_mat.sum(axis=1)).mean()
  # uaprc
  max_probs = probs.max(axis=1)
  correctness_labels = (preds == labels)
  auprc, _, _, _ = pr_curve(correctness_labels, max_probs, pos_label=1)
  
  return probs, preds, labels, acc, uar, conf_mat, auprc, acc_at_3, acc_at_5

def accuracy_at_k(logits_or_probs, labels, k):
  sorted_preds = np.argsort(logits_or_probs, axis=1)
  k_new = min(k, logits_or_probs.shape[-1])
  return np.any(sorted_preds[:, -k_new:] == labels.reshape([len(labels), 
                                                            1]), 
                axis=1).mean()

def softmax_numpy(logits, temperature=1.0, simple_dataset=np.e):
  if simple_dataset == np.e:
    step1 = np.exp(logits / temperature)
  else:
    step1 = np.power(simple_dataset, logits / temperature)
  step2 = step1.sum(axis=1)
  step3 = step1 / step2.reshape([len(step2), 1])
  return step3

def sigmoid_numpy(x):
  return 1 / (1 + np.exp(-x))

def confusion_matrix_np(preds, labels, normalized=True):
  n_classes = int(labels.max() + 1)
  mat = np.zeros([n_classes, n_classes])  # actual x predicted
  for i in range(0, n_classes):
    actual_mask = (labels == i)
    for j in range(0, n_classes):
      predicted_mask = (preds == j) 
      mat[i,j] = np.logical_and(actual_mask, predicted_mask).sum()
      
  # Make sure no errors of funny stuff
  assert mat.sum() == len(preds)
  
  # Normalize
  if normalized:
    mat = mat / float(len(preds))
    
  return mat

def calc_acc_numpy(is_train, logits, labels, test_samples, temperature=1.0):
  assert is_train == False
  probs = softmax_numpy(logits, temperature)
  probs = average_or_max_tensor_numpy(probs, test_samples)
  preds = np.argmax(probs, axis=1)
  labels = downsample_labels_numpy(labels, test_samples)
  return (preds == labels).mean()

def calc_UAR_numpy(is_train, logits, labels, test_samples, temperature=1.0):
  assert is_train == False
  probs = softmax_numpy(logits, temperature)
  probs = average_or_max_tensor_numpy(probs, test_samples)
  preds = np.argmax(probs, axis=1)
  labels = downsample_labels_numpy(labels, test_samples)
  n_classses = int(labels.max() + 1)
  s = 0
  for i in range(0, n_classses):
    from_class_mask = (labels == i)
    preds_from_class = preds[from_class_mask]
    labels_from_class = labels[from_class_mask]
    class_acc = (preds_from_class == labels_from_class).mean()
    s += class_acc
  return s / float(n_classses)

def pr_curve(binary_labels, scores, pos_label=1):
  precision, recall, thresholds = precision_recall_curve(binary_labels, scores, pos_label)
  auc_result = auc(recall, precision)
  return auc_result, precision, recall, thresholds

def average_or_max_tensor_numpy(tensor, test_samples, op='mean'):
  assert tensor.shape[0] % test_samples == 0
  n_examples = tensor.shape[0] / test_samples
  n_classes = tensor.shape[1]
  resh = tensor.reshape([n_examples, test_samples, n_classes])
  if op == 'mean':
    return resh.mean(axis=1)
  elif op == 'max':
    return resh.max(axis=1)
 
def downsample_labels_numpy(labels, test_samples):
  return labels[::test_samples]


######################################## TensorFlow based evaluations #######################################

# Two reshaping functions, used for calculating the monitoring channels on a test batch with multiple samples per example
def average_logits(t, test_mb, test_samples):
    last_dim = t.get_shape()[-1].value
    reshaped = tf.reshape(t, [test_mb, test_samples, last_dim])
    return tf.reduce_mean(reshaped, 1)
# Reshaping for the labels
def downsample_labels(t, test_mb, test_samples):
  reshaped = tf.reshape(t, [test_mb, test_samples])
  return reshaped[:, 0]

def classification_accuracy(is_train, logits, labels, test_mb=0, test_samples=0):
  assert is_train
  if is_train:
    return tf.reduce_mean(tf.to_float(tf.nn.in_top_k(logits, labels, 1)))
  else:
    assert 1==2
    avgd_logits = average_logits(logits, test_mb, test_samples)
    downsampled_labels = downsample_labels(labels, test_mb, test_samples)
    return tf.reduce_sum(tf.to_float(tf.nn.in_top_k(avgd_logits, downsampled_labels, 1)))