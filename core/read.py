from __future__ import division, absolute_import, print_function
import os, tensorflow as tf, json, io
from .common import get_debug_session
from six import iteritems, string_types

FLAGS = tf.app.flags.FLAGS


def get_files(files_list):
  """
  Gets the list of files from the 'files_list' file.

  Args:
    files_list: path to text file with the files list

  Returns:
    list of file names and the size of the files
  """
  filenames = []
  lengths = []
  relative_dir = os.path.dirname(files_list)
  f = open(files_list, 'r')
  lines = f.readlines()
  for line in lines:
    if not (line.startswith('#') or line == ''):
      fields = line.split()
      filename = fields[0]
      length = fields[1] if len(fields) > 1 else 0
      filenames.append(os.path.os.path.join(relative_dir, filename))
      lengths.append(int(length))

  return filenames, lengths

def get_tfr_structure_from_file(tfr_structure_file):
  with io.open(tfr_structure_file, encoding='utf8') as json_data:
    tfr_structure = json.load(json_data)
  return tfr_structure


def read_and_decode(tfr_structure, filename_queue):
  """
  Design of the tfr_structure:
  The tfr_structure is a dictionary that is used to decode the .tfr files back into tensors of the correct names and
  shapes. It is saved as a .json (for easy editing) files usually in the dataset's home dir.
  The tfr_structure dict is comprised of (key: value) pairs, where:
  - key: determines the name of the tensor that will be added to the endpoints
  - value is [name of field in the .tfr file, type, shape],
      - usually 'name of field in the .tfr file' = 'key'
      - shape is a list [dim_1, ... , dim_n] where dim_i can either be simply an integer (when it is fixes across the
      dataset, or a string (when it varies in the dataset, such as the length of audio file) that is the name of the
      field in the .tfr file containing the actual value.

  Design of .tfr files:
  Each field in the .tfr file is a tensor. These fields can be from one of two groups (accrording to the explanation
  above):
  - tensors that will end up as an endpoint (see 'name of field in the .tfr file' above). Can be multidimensional.
  - tensors that denote shapes of other tensors from the first group. Always int.
  A tensor that is multidimensional, will be converted to string (when creating the .tfr file) and appear in the .tfr
  file as string. Using the tfr_structure, we can reconvert the .tfr file into the correct tensors.
  """
  # Read from the queue
  reader = tf.TFRecordReader()
  filename, serialized_example = reader.read(filename_queue)
  
  # tfr file to tensor
  features = {}
  # At the first stage, extract all tensors from the .tfr file (both tensors that will be endpoints, and tensors that denote
  # shapes of other tensors.
  # if the shape is not [], this is a sequence/matrix and we store it as bytes, therefore the type is string 
  for k,v in iteritems(tfr_structure):
    features_name, dtype, shp = v
    # This is not a sequence / matrix
    if shp == [] and dtype != 'string':
      if dtype == 'int':
        features[features_name] = tf.FixedLenFeature([], tf.int64)
    # This is a sequence / matrix
    else:
      features[features_name] = tf.FixedLenFeature([], tf.string)
      for x in shp:
        if isinstance(x, string_types):
          features[x] = tf.FixedLenFeature([], tf.int64)
  tensors_pre_cast = tf.parse_single_example(serialized_example, features=features)
  
  # At the second stage, we convert each tensor back to its correct datatype and shape, and create the endpoints. 
  tensors_post_cast = {}
  for k,v in iteritems(tfr_structure):
    features_name, dtype, shp = v
    # This is not a sequence / matrix
    if dtype == 'string':
      tensors_post_cast[k] = tensors_pre_cast[k]
    if shp == []:
      if dtype == 'int':
        tensors_post_cast[k] = tf.cast(tensors_pre_cast[features_name], tf.int32)
    # This is a sequence / matrix 
    else:
      if dtype == 'int':
        seq_features = tf.decode_raw(tensors_pre_cast[features_name], tf.int32)
      elif dtype == 'float':
        seq_features = tf.decode_raw(tensors_pre_cast[features_name], tf.float32)
      elif dtype == 'uint8':
        seq_features = tf.decode_raw(tensors_pre_cast[features_name], tf.uint8)
      else:
        raise ValueError
      seq_shp = []
      for x in shp:
        if isinstance(x, string_types):
          seq_shp.append(tf.cast(tensors_pre_cast[x], tf.int32))
        else:
          seq_shp.append(x)
      tensors_post_cast[k] = tf.reshape(seq_features, tf.stack(seq_shp))
      
  # Add the filename
  tensors_post_cast['filename'] = filename
    
  return tensors_post_cast


def get_input(is_train, app, sess):
  """
  Design:
  This function is responsible for reading the .tfr files and creating a queue from which the neural network dequeue
  batches. There are three main types of operations to distinct:
  a) (ONCE): Operations that are performed only once, at data-loading time, before training begins.
  b) (CPU): Operations that are performed repeatedly and on the fly during training, using multiple parallel cpu
  threads.
  c) (GPU): The neural network operations, performed on GPU while training.

  Without cache:
  The highlevel workflow is: [Read from disk --> preprocess](CPU) --> [neural network](GPU).
  The "Read from disk --> preprocess" part will be executed every time the network wants to process a batch.
  When using a neural network normally we feed the same data over and over to the network, so this is not very
  efficient and often cause performance bottlenecks (since the GPU waits for the hard-drive to load the data
  and for the CPU for preprocess it).
  The functions to implement here are precache_processing, postcache_processing: each gets and returns the a dictionary
  of tensors. These two functions will be called one after the other, so it doesn't matter which operation is in which
  one of them. This is implemented as follows:
  - read a .tfr file, apply precache and postcache preprocessing, feed it to the batches queue, from which the network
  dequeues minibatches. All is happening repeatedly and on the fly.

  With cache:
  The high-level workflow is:
  [Read from disk --> precache preprocess --> examples selection --> cache](ONCE) --> [postcache preprocessing](CPU) -->
  [neural network](GPU).
  In this setting, which is more recommended for experiments with GPUs, we allow some preprocessing and examples
  selection before caching the examples.
  This is implemented as following:
  - read all .tfr files from disk, apply precache preprocessing, have a list of all examples in a numpy fortmat:
  'all_examples_npy'.
  - apply examples selection (the function 'dataset_level_processing') if defined. Chosen examples in numpy format
  will be in 'chosen_examples_npy'.
  - Create an examples queue (this is the cache itself) and fill it with all the examples 'chosen_examples_npy'.
  - Link the enqueue and dequeue ops for this queue, so when we dequeue an example from it we immediately enqueue in
  back.
  - The following part is the only which will be run on repeatedly and on the fly: Enqueue an example from the examples
  queue (the cache), apply postcache preprocessing, and enqueue into a new batches queue, from which the network
  dequeues minibatches.

  * precache preprocessing takes tensors that contain each one example (the same one), and returns the same
  * postcahce preprocessing also takes tensors that contain each one example (the same one), for the train graph,
    it also returns one example, but for the eval graph, it must prepend an additional batch axis to each tensor, and
    it can return multiple examples.
  """
  with tf.device('/cpu:0'):
    # Set the correct values and functions to work with
    cl = app['app']
    files_list = cl.train_files if is_train else cl.eval_files
    cache = app['cache']
    filenames_queue_func = cl.create_filename_queue
    dataset_level_processing = cl.dataset_level_processing if hasattr(cl, 'dataset_level_processing') else None
    assert not (dataset_level_processing and not cache)
    precache_processing = cl.precache_processing
    postcache_processing = cl.postcache_processing
    n_threads = 16 if is_train else 1
    tfr_structure_file = cl.tfr_structure_file
    mb = app['train_mb'] if is_train else tf.placeholder(dtype=tf.int32, 
                                                       shape=[], 
                                                       name='{}_test_mb'.format(app['app'].name))
    
    # Get the files metadata
    files, lengths = get_files(files_list)
    tfr_structure = get_tfr_structure_from_file(tfr_structure_file)

    if not cache:
      n_examples = len(files)
      filename_queue = filenames_queue_func(is_train, files, lengths)
      
      # Read the example
      val_dict = read_and_decode(tfr_structure, filename_queue)
      
      # Precache and postcache processing
      val_dict = precache_processing(is_train, val_dict)
      val_dict = postcache_processing(is_train, val_dict, tfr_structure)
      if not is_train:
        # fill the queue
        sess.run(filename_queue.enqueue_many(tf.constant(files)))
        # overwrite n_examples with the real number
        n_examples = get_example_number_after_postprocessing(sess, val_dict, n_examples)
      
      # Push to the batches queue
      return n_examples, tf.train.batch(
        val_dict,
        mb,
        num_threads=n_threads, 
        capacity=3000, 
        enqueue_many=not is_train)
      
    else:    
      # Create and fill the filenames queue
      # Random Uniformly across files
      filename_queue = tf.FIFOQueue(capacity=len(files), dtypes=tf.string)
      fill_filename_queue = filename_queue.enqueue_many(tf.constant(files))
      sess.run(fill_filename_queue)
      
      # Tensors for reading one example from the disk
      tfr_dict = read_and_decode(tfr_structure, filename_queue)
      
      # Perform a per-example precache preprocessing
      precache_dict = precache_processing(is_train, tfr_dict)
      
      # Load all examples from disk
      print('loading examples from disk')
      all_examples_npy = []
      for j in range(0, len(files)):
        all_examples_npy.append(sess.run(precache_dict))

      # Perform preprocessing on a whole dataset level. 
      # Should not change shapes or types of tensors, nor add/remove tensors.
      if dataset_level_processing:
        chosen_examples_npy = dataset_level_processing(is_train, all_examples_npy)
        all_examples_npy = None
      else:
        chosen_examples_npy = all_examples_npy
      n_examples = len(chosen_examples_npy)
      
      # Create a queue to be filled with cached examples and its enqueue op     
      if is_train:
        examples_queue = tf.RandomShuffleQueue(
          capacity=n_examples, 
          dtypes=[val.dtype for key, val in iteritems(precache_dict)],
          names=[key for key, val in iteritems(precache_dict)],
          shapes=None,
          min_after_dequeue=0)
      else:
        examples_queue = tf.FIFOQueue(
          capacity=n_examples, 
          dtypes=[val.dtype for key, val in iteritems(precache_dict)],
          names=[key for key, val in iteritems(precache_dict)],
          shapes=None)
      example_enq_op = examples_queue.enqueue(precache_dict)
      
      # Fill the examples queue
      for j in range(0, n_examples):
        sess.run(example_enq_op, feed_dict={precache_dict[k]: chosen_examples_npy[j][k] for k in precache_dict.keys()})
      
      # Connect the dequeue and enqueue ops
      intermediate_dict = examples_queue.dequeue()
      enq_back = examples_queue.enqueue(intermediate_dict)
      with tf.control_dependencies([enq_back]):
        for key in intermediate_dict.keys():
          intermediate_dict[key] = tf.identity(intermediate_dict[key])
          
      # As the queue forgets about shapes if they are not fully defined / not set through the 'shapes' param,
      # We set the shapes back
      for k in precache_dict.keys():
        intermediate_dict[k].set_shape(precache_dict[k].shape)
      
      # Perform the postcache processing
      postcache_dict = postcache_processing(is_train, intermediate_dict, tfr_structure)

      # Finding the number of evaluation examples
      if not is_train:
        # overwrite n_examples with the real number
        n_examples = get_example_number_after_postprocessing(sess, postcache_dict, n_examples)
        print(n_examples)

      # Create a batches queue
      batch_queue = tf.FIFOQueue(
        capacity=3000,
        dtypes=[val.dtype for key, val in iteritems(postcache_dict)],
        names=[key for key, val in iteritems(postcache_dict)],
        shapes=[val.get_shape() if is_train else val.get_shape()[1:] for key, val in iteritems(postcache_dict)])
      final_enq = batch_queue.enqueue(postcache_dict) if is_train else batch_queue.enqueue_many(postcache_dict)
      qr = tf.train.QueueRunner(batch_queue, [final_enq] * n_threads)
      tf.train.add_queue_runner(qr, tf.GraphKeys.QUEUE_RUNNERS)
      return n_examples, batch_queue.dequeue_many(mb)


def get_example_number_after_postprocessing(sess, postcache_tensor, n_examples_pre_postprocessing):
  n_examples_postprocessing = 0
  for j in range(0, n_examples_pre_postprocessing):
    postcache_numpy = sess.run(postcache_tensor)
    # take one key, doesn't matter which (hopefully!)
    key = list(postcache_numpy.keys())[0]
    n_examples_postprocessing += postcache_numpy[key].shape[0]
  return n_examples_postprocessing
