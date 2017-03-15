import tensorflow as tf
from .blocks import gather_nd

tf.app.flags.DEFINE_float('alpha', 1.0, '')
tf.app.flags.DEFINE_float('beta', 1.0, '')

FLAGS = tf.app.flags.FLAGS


def ce_loss(logits, labels, **kwargs):
  """
  The standard classification loss. Applies softmax on the logits and computes the loss.
  """
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
  return tf.reduce_mean(cross_entropy)


def ce_loss_from_softmax(softmax_out, labels, avg=True, **kwargs):
  """
  The standard classification loss. Takes the softmax output and computes the loss.
  """
  indices = tf.transpose(tf.stack([tf.constant(range(0, softmax_out.get_shape()[0].value)), labels]), [1,0])
  correct_probs = gather_nd(softmax_out, indices)
  loss = -tf.reduce_mean(tf.log(correct_probs)) if avg else -tf.log(correct_probs)
  return loss


def binary_ce_loss(logits, labels, n_classes, **kwargs):
  """
  Binary CE loss, for multilabel classification and other applications.
  """
  one_hot_labels = tf.one_hot(labels, n_classes)
  loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
  return tf.reduce_mean(loss)


def MSE_loss(tensor, targets):
  """
  Standard MSE loss.
  """
  loss = tf.reduce_mean(tf.pow(tensor - targets, 2))
  return loss


def mix_sigmoid_ce_loss(logits, labels, n_classes, **kwargs):
  """
  A mix between the standard CE and binary CE loss, according to alpha and beta.
  """
  print('alpha, beta:', FLAGS.alpha, FLAGS.beta)
  loss = ce_loss(logits, labels) * FLAGS.alpha +  binary_ce_loss(logits, labels, n_classes) * FLAGS.beta
  return loss
