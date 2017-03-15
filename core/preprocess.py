import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def audio_filename_queue(is_train, files, lengths):
  return tf.train.string_input_producer(files, capacity=320)
  
def normalize_each_feature(features, axes=[0]):
  mean, variance = tf.nn.moments(features, axes=axes, keep_dims=True)
  variance_no_zero = tf.where(tf.greater(variance, 0), 
                               variance, 
                               tf.ones(tf.shape(variance), 
                                       dtype=variance.dtype))
  return (features - mean) / tf.sqrt(variance_no_zero)

def zero_padding(audio_features, target_length):
  # Pad with zeros to match the sample length
  l = target_length
  if audio_features.get_shape().as_list()[0]:
    length = audio_features.get_shape()[0].value
    padding = [[0, max(0, l - length)],
               [0,0], 
               [0,0]]
  else:
    dynamic_shape = tf.shape(audio_features) 
    length = dynamic_shape[0]
    padding = tf.stack([tf.stack([0,
                                tf.maximum(0, l - length)]), 
                       [0,0], 
                       [0,0]])
  audio_features_padded = tf.pad(audio_features, padding)
  
  return audio_features_padded

def crop(is_train, tensor, test_samples, target_length):
  n_features = tf.shape(tensor)[1]
  if is_train:
    audio_features_cropped = random_crop_with_stride(tensor, 
                                                    [target_length, n_features, 1],
                                                    [FLAGS.crop_strides, 1, 1])
  else:
    crops = []
    dynamic_shape = tf.shape(tensor)
    length = dynamic_shape[0]
    last_possible = length - target_length
    if test_samples > 1:
      slice = last_possible // (test_samples - 1)
    else:
      slice = 0
    for i in range(0, test_samples):
      start = slice * i
      crops.append(tf.slice(tensor, 
                            [start, 0, 0], 
                            [target_length, n_features, 1]))
    audio_features_cropped = tf.stack(crops)
    
  return audio_features_cropped

def random_crop_with_stride(value, size, strides, seed=None):
    shape = tf.shape(value)
    limit = ((shape - size) / strides) + 1
    limit = tf.to_int32(limit)
    offset = tf.random_uniform(tf.shape(shape), 
                               dtype=tf.int32,
                               maxval=tf.int32.max, 
                               seed=seed) % limit
    offset = offset * strides
    return tf.slice(value, offset, size)
  
def pad_1D_for_windowing(tensor, length):
  """
  Pads a tensor on the first dimension, from both sides, to allow a windows of size 'length' to be centered
  in each element of the first dimension
  """
  len_before = ((length + 1) // 2) - 1
  len_after = length // 2
  return tf.pad(tensor, 
                [[len_before, len_after]] + [[0,0]] * (len(tensor.shape) - 1))
  
def synchronized_1D_crop(tensors, length, stride, padding, seed=None):
  """
  Tensors t should have the same t.shape[0].
  """
  # Pad if needed. If padding=True, we will pad from both sides, to allow cropped patches that are centered
  # In every element of the original tensor.
  if padding:
    padded_tensors = []
    for t in tensors:
      padded_tensors.append(pad_1D_for_windowing(t, length))
  else:
    padded_tensors = tensors
    
  # Calculate a random offset to crop from  
  limit = ((tf.shape(padded_tensors[0])[0] - length) / stride) + 1
  limit = tf.to_int32(limit)
  offset = tf.random_uniform([], 
                             dtype=tf.int32,
                             maxval=tf.int32.max, 
                             seed=seed) % limit
  offset = offset * stride
  
  # Do the cropping
  cropped = [t[offset:offset+length] for t in padded_tensors]

  # Reshape to have fully defined shapes 
  return [tf.reshape(t, [length] + t.shape.as_list()[1:]) for t in cropped]

def expand_dims_to_match(ref_tensor, tensor_to_expand):
  out = tf.expand_dims(tensor_to_expand, 0)
  return tf.tile(out, 
                 [tf.shape(ref_tensor)[0]] + [1]*tensor_to_expand.shape.ndims)

def expand_first_dim(d):
  for k in d.keys():
    d[k] = tf.expand_dims(d[k], 0)
