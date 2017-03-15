# Import from standard libraries
from __future__ import division, absolute_import, print_function
import tensorflow as tf, os, time, numpy as np, importlib
from six import iteritems
from datetime import datetime

# Import from our libraries
from . import models
from .common import SUMMARIES_COLLECTION, BN_COLLECTION, make_summaries, get_debug_session, SUMMARY_DIR, CHECKPOINT_DIR, MOVING_AVG_COLLECTION, Channel, NON_TRAIN_VARS, NON_INIT_VARS
from .read import get_input
from .models import common_first_layers

# Flags
FLAGS = tf.app.flags.FLAGS

def prepare_train(apps, inference, sess, restore_path, model_name):  
  # Splits a tensor back to the different modules
  def split_to_apps(tensor, apps, key, in_endpoints=False):
    l = len(tensor.get_shape()) - 1
    r = tensor.get_shape().as_list()[1:]
    s = 0
    for m in apps:
      to_set = tf.slice(tensor, [s] + [0]*l, [m['train_mb']] + r)
      if in_endpoints:
        m['endpoints'][key] = to_set
      else:
        m[key] = to_set 
      s += m['train_mb']
  
  # Feed each dataset to its dedicated first layers
  for m in apps:
    if hasattr(m['app'], 'first_layers'):
      m['app'].first_layers(True, m['endpoints'])
    else:
      common_first_layers(True, m['endpoints'], m['app'].name)
  
  # Feed through the bottleneck and write down the variables so far
  preprocess_vars = tf.global_variables()
  global_endpoints = inference(True, 
                               sess, 
                               [m['endpoints'] for m in apps])
  bottleneck_vars = list(set(tf.global_variables()) - set(preprocess_vars))
  
  # Split back after the bottleneck
  split_to_apps(global_endpoints['bottleneck'], 
                apps, 
                'bottleneck', 
                in_endpoints=True)
  
  # Get the logits, losses, channels, for each dataset. Losses might be lists, channels is a dict. 
  # All losses will be added to the channels
  for m in apps:
    if hasattr(m['app'], 'top_layers'):
      m['app'].top_layers(True, global_endpoints, m['endpoints'])
    m['losses'] = m['app'].compute_loss(global_endpoints, m['endpoints'])
    m['channels'] = m['app'].monitoring_channels(True, global_endpoints, m['endpoints'])
  
  # Init the channels dict
  channels = {}
  
  # Gather endpoints, losses and channels
  for m in apps:
    for i, l in enumerate(m['losses']):
      tf.add_to_collection(tf.GraphKeys.LOSSES, l)
      channels[m['app'].name + '_loss_' + str(i)] = l
      tf.summary.scalar(m['app'].name + '_loss_' + str(i), l)
    channels.update(m['channels'])
  losses = tf.get_collection(tf.GraphKeys.LOSSES)
  loss = tf.add_n(losses, name='loss') / len(losses)
  channels['loss'] = loss
  tf.summary.scalar('loss', loss)
  
  # Add channels and summaries for the average accuracy and UAR
  accuracies = []
  UARs = []
  for key, item in iteritems(channels):
    if key.endswith('accuracy'):
      accuracies.append(item)
    if key.endswith('UAR'):
      UARs.append(item)
  if len(accuracies) > 0:
    val = tf.add_n(accuracies) / len(accuracies)
    channels['average_accuracy'] = val
    tf.summary.scalar('average_accuracy', val)
  if len(UARs) > 0:
    val = tf.add_n(UARs) / len(UARs)
    channels['average_UAR'] = val
    tf.summary.scalar('average_UAR', val)
  
  # The global step
  global_step = tf.Variable(0, trainable=False)
  
  # Learning rate decay
  if FLAGS.decay_steps > 0:
    lr = tf.train.exponential_decay(FLAGS.lr, global_step, FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)
  else:
    lr = FLAGS.lr
  tf.summary.scalar('lr', lr)
  
  # Create the training algorithm
  print('Creating the optimization graph nodes')
  if FLAGS.alg == 'sgd':
    opt = tf.train.GradientDescentOptimizer(FLAGS.lr)
  elif FLAGS.alg == 'momentum':
    opt = tf.train.MomentumOptimizer(FLAGS.lr, FLAGS.mom)
  elif FLAGS.alg == 'rmsprop':
    opt = tf.train.RMSPropOptimizer(FLAGS.lr, momentum=FLAGS.mom)
  elif FLAGS.alg == 'adadelta':
    opt = tf.train.AdadeltaOptimizer(FLAGS.lr)
  elif FLAGS.alg == 'adagrad':
    opt = tf.train.AdagradOptimizer(FLAGS.lr)
  elif FLAGS.alg == 'adam':
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
  else:
    raise ValueError("Unknown algorithm")

  # Set the variables to train
  vars_to_train = set(tf.trainable_variables()) - set(tf.get_collection(NON_TRAIN_VARS))
  if restore_path and not FLAGS.finetune_all:
    vars_to_train = vars_to_train - set(bottleneck_vars)
      
  # Compute and apply gradients
  grads_no_clipping = opt.compute_gradients(loss=loss, var_list=list(vars_to_train))
  flat = tf.concat(axis=0, values=[tf.reshape(g[0], [-1]) for g in grads_no_clipping])
  avg = tf.reduce_mean(tf.abs(flat))
  tf.summary.scalar('gradient_size', avg)
  channels['gradient_size'] = avg
  if FLAGS.monitor_grad_norms:
    for g in grads_no_clipping:
      norm = tf.sqrt(tf.reduce_sum(g[0]*g[0]))
      channels[g[1].name + '_grad_norm'] = norm
      tf.summary.scalar(g[1].name + '_grad_norm', norm)
  if FLAGS.gc > 0:
    grads = []
    for g in grads_no_clipping:
      grads.append((tf.clip_by_norm(g[0], FLAGS.gc), g[1]))
  else:
    grads = grads_no_clipping
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  print('num of all variables is {}'.format(sum([x.get_shape().num_elements() for x in tf.global_variables()])))
  print('num of trainable variables is {}'.format(sum([x.get_shape().num_elements() for x in vars_to_train])))
  
  # Add summaries for all gradients and variables
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)
  for grad, var in grads:
    if grad is not None:
      tf.add_to_collection(SUMMARIES_COLLECTION, (var.op.name + '/gradients', grad))
  make_summaries()
  
  # Add summaries for the number of elements in the queues
  for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
    tf.summary.scalar(qr.queue.name, qr.queue.size())
  
  # BN
  batchnorm_updates = tf.get_collection(BN_COLLECTION)
  batchnorm_updates_op = tf.group(*batchnorm_updates)
  
  # Moving averages. Every element in the collection MOVING_AVG_COLLECTION is a tuple of (var, name).
  emv = tf.train.ExponentialMovingAverage(0.95)
  moving_avg_op = emv.apply([x[0] for x in tf.get_collection(MOVING_AVG_COLLECTION)])
  for x in tf.get_collection(MOVING_AVG_COLLECTION):
    tf.summary.scalar(x[1], emv.average(x[0]))
  
  # Grouping all ops needed for training
  train_op = tf.group(apply_gradient_op, batchnorm_updates_op, moving_avg_op)

  # Restore variables
  if restore_path:
    # Restore the variables values
    print('Restoring model from {}'.format(restore_path))
    vars_to_restore = tf.global_variables() if FLAGS.restore_all else bottleneck_vars  
    restore_saver = tf.train.Saver(vars_to_restore)
    restore_saver.restore(sess, restore_path)
  else:
    vars_to_restore = []
  
  # Init
  init_vars = set(tf.global_variables()) - set(tf.get_collection(NON_INIT_VARS)) - set(vars_to_restore)
  init_op = tf.variables_initializer(list(init_vars))
  sess.run(init_op)
      
  # Saver and summary
  saver = tf.train.Saver(tf.global_variables())
  summary_op = tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter(SUMMARY_DIR + '/' + str(model_name) + '/train', tf.get_default_graph())

  return channels, saver, summary_writer, summary_op, train_op, global_endpoints


def prepare_evaluation(apps, inference, model_name):
  # Feed through the preprocessing layers and the bottleneck
  for m in apps:
    with m['graph'].as_default():
      if hasattr(m['app'], 'first_layers'):
        m['app'].first_layers(False, m['endpoints'])
      else:
        common_first_layers(False, m['endpoints'], m['app'].name)
      m['global_endpoints'] = inference(False, m['sess'], [m['endpoints']])
      m['endpoints']['bottleneck'] = m['global_endpoints']['bottleneck']
    
  # Get the logits, and channels, for each dataset. Losses might be lists, channels is a dict.
  for m in apps:
    with m['graph'].as_default():
      if hasattr(m['app'], 'top_layers'):
        m['app'].top_layers(False, m['global_endpoints'], m['endpoints'])
      m['channels'] = m['app'].monitoring_channels(False, m['global_endpoints'], m['endpoints'])
  
  # Gather global endpoints
  global_endpoints = {}
  for m in apps:
    global_endpoints.update(m['global_endpoints'])
  
  # Create savers that will restore variables, one for each dataset (because the graph is different)
  for m in apps:
    with m['graph'].as_default():
      m['saver'] = tf.train.Saver(tf.global_variables())

  # Build the summary operation based on the TF collection of summaries.
  summary_writer = tf.summary.FileWriter(SUMMARY_DIR + '/' + str(model_name) + '/test', tf.get_default_graph())

  return summary_writer, global_endpoints


def eval_once(apps, summary_writer, model_name):
  """
  Evaluates each dataset separately.
  """
  # Get the summary to add some values to
  summary = tf.Summary()

  # Print the header
  print('----------------- TEST MONITOR ----------------------')
  
  # Find the right checkpoint
  ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR, latest_filename=model_name)
  if ckpt and ckpt.model_checkpoint_path:
    print('Restoring from file: {}'.format(ckpt.model_checkpoint_path))
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
  else:
    print('No checkpoint file found')
    return
  
  # lists for the channels we accumulate across tasks
  accuracies = []
  UARs = []
  domain_accs = []
  all_channels = {}
  
  for m in apps:
    with m['graph'].as_default():
      # restore from checkpoint
      sess = m['sess']
      m['saver'].restore(sess, ckpt.model_checkpoint_path)

      # this number doesn't really have to be set by the applications because it doesn't affect much
      test_mb = m['app'].test_mb
      # Number of examples and iterations
      num_iter = m['n_examples'] // test_mb # this is the integer division. it should work this way
      examples_for_last_batch = m['n_examples'] % test_mb
      total_sample_count = num_iter * test_mb
  
      # Initialize aggregators
      concate_aggregators = {}
      for k,v in iteritems(m['channels']):
        if isinstance(v, Channel):
          concate_aggregators[k] = None
      print_aggregators = {}
      for k,v in iteritems(m['channels']):
        assert k != 'filenames'   # Just to make sure no errors will happen
        if (isinstance(v, Channel) and v.printable) or (not isinstance(v, Channel)):
          print_aggregators[k] = 0.0
      
      # Create the tensor_channels
      tensor_channels = {}
      for k,v in iteritems(m['channels']):
        if isinstance(v, Channel):
          tensor_channels[k] = v.tensor
        else:
          tensor_channels[k] = v
      
      step = 0
      while step < (num_iter + 1):  ###tag:Queues###
        if step == num_iter:
          # in the last step the batch size is different
          this_mb = examples_for_last_batch          
        else:
          this_mb = test_mb
        step += 1
        
        # Skip the last iteration if m['n_examples'] % test_mb == 0
        if this_mb == 0:
          continue

        # Get the monitoring channels for one batch
        results = sess.run(tensor_channels, 
                           feed_dict={'{}_test_mb:0'.format(m['app'].name): this_mb})

        # Sum the train monitoring channels
        for k in print_aggregators.keys():
          print_aggregators[k] += results[k]
        
        # concatenation aggregation of certain channels
        for k in concate_aggregators.keys():
          if step == 1:
            concate_aggregators[k] = [results[k]]
          else:
            concate_aggregators[k].append(results[k]) #np.concatenate([concate_aggregators[k], results[k]])
          
      for k in concate_aggregators.keys():
        concate_aggregators[k] = np.concatenate(concate_aggregators[k])
      
      # Dump activations
      if FLAGS.dump_activations:
        for k in concate_aggregators.keys():
          np.save('{}_{}'.format(model_name, k), concate_aggregators[k])
      
      # Get the numpy test channels
      if hasattr(m['app'], 'numpy_channels'):
        print_aggregators.update(m['app'].numpy_channels(concate_aggregators, int(global_step)))

      all_channels.update(print_aggregators)

      # Do some printing and summary
      for k, v in iteritems(print_aggregators):
        # Some channels we aggregate over tasks we don't want to print. The rest we do
        print('{}_{}: {}'.format(m['app'].name, k, v))
        if type(v) != np.ndarray:
          summary.value.add(tag='{}_{}'.format(m['app'].name, k), simple_value=v)

      # Collect the channels we want to aggregate over different tasks
      for k, v in iteritems(print_aggregators):
        if k.endswith('accuracy'):
          accuracies.append(v)
        if k.endswith('UAR'):
          UARs.append(v)
        if k == 'domain_classification_rate':
          domain_accs.append(v)
          
  # Print the channels we accumulate across tasks
  if len(accuracies) > 0:
    val = sum(accuracies) / len(accuracies)
    print('mean accuracy: {}'.format(val))
    summary.value.add(tag='mean_accuracy', simple_value=val)
  if len(UARs) > 0:
    val = sum(UARs) / len(UARs)
    print('mean UAR: {}'.format(val))
    summary.value.add(tag='mean_UAR', simple_value=val)
  if len(domain_accs) > 0:
    val = sum(domain_accs) / len(domain_accs)
    print('mean domain accuracy: {}'.format(val))
    summary.value.add(tag='domain accuracy', simple_value=val)

  # Print the bottom part
  print('-----------------------------------------------------')
  print('')

  # Write the summary, after we already added stuff to it
  summary_writer.add_summary(summary, global_step)
  
  return all_channels


def main_loop(train_apps, eval_apps, action_manager):
  # Print FLAGS
  print('----------------------------- FLAGS VALUES --------------------------------')
  for k in sorted(FLAGS.__flags.keys()):
    print('{}: {}'.format(k, FLAGS.__flags[k]))
  
  # Print General model information
  print('----------------------- DATA LOADING, MODEL PREPARING -------------------------')
  time_stamp = str(datetime.now()).replace(':', '-').replace(' ', '_')
  model_name = FLAGS.model_name if FLAGS.model_name else str(time_stamp)
  print('model_name: {}'.format(model_name))
  
  # Print minibatches size
  for m in train_apps:
    print('train_mb for dataset {}: {}'.format(m['app'].name, m['train_mb']))
    
  # The bottleneck function.
  # If contains '.', we load it from a full path, otherwise we load it from the models file. 
  if '.' not in FLAGS.inference_bottleneck: 
    inference_bottleneck = getattr(models, FLAGS.inference_bottleneck)
  else:
    sep_index = FLAGS.inference_bottleneck.rindex('.')
    containing_module = importlib.import_module(FLAGS.inference_bottleneck[:sep_index])
    inference_bottleneck = getattr(containing_module, FLAGS.inference_bottleneck[sep_index+1:])
  
  # Train model
  t_g = tf.Graph()
  with t_g.as_default():
    # Create the session
    sess = tf.Session(graph=t_g,      
                      config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement, 
                                            allow_soft_placement=True))
    
    # Get the input for all datasets
    for m in train_apps:
      with tf.name_scope('input/{}'.format(m['app'].name)):
        if hasattr(m['app'], 'get_input'):
          get_input_func = m['app'].get_input
        else:
          get_input_func = get_input
        m['n_examples'], m['endpoints'] = get_input_func(is_train=True, 
                                                         app=m, 
                                                         sess=sess)
      print('Training clips for dataset {}: {}'.format(m['app'].name, 
                                                       m['n_examples']))

    # Construct the training graph and make training preparations
    with tf.device('/gpu:' + FLAGS.gpu):
      t_channels, t_saver, t_summary_writer, t_summary_op, train_op, t_endpoints = prepare_train(train_apps,
                                                                                                inference_bottleneck, 
                                                                                                sess, 
                                                                                                FLAGS.restore_path, 
                                                                                                model_name)
  
  # Eval model
  # Get the input for all datasets
  for m in eval_apps:
    g = tf.Graph() 
    with g.as_default():
      # Create the session for each eval graph
      m['sess'] = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement,
                                                   allow_soft_placement=True))
      
      # Get the input for all datasets
      if hasattr(m['app'], 'get_input'):
        get_input_func = m['app'].get_input
      else:
        get_input_func = get_input
      m['n_examples'], m['endpoints'] = get_input_func(is_train=False, 
                                                       app=m, 
                                                       sess=m['sess']) 
      print('Eval clips for dataset {}: {}'.format(m['app'].name, 
                                                   m['n_examples']))
      m['graph'] = g
  best_step, best_test_channels = (0, {})
    
  # Construct the eval graph and make other preparations
  with tf.device('/gpu:' + FLAGS.gpu):
    e_summary_writer, e_endpoints = prepare_evaluation(eval_apps, 
                                                       inference_bottleneck, 
                                                       model_name)
      
  with t_g.as_default():  # Not sure why needed, works without it on local machine, but not on sgesabatos
    # Start all the threads that fill the queues (only if reading data with queues) ###tag:Queues###
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    szs = [x.queue.size() for x in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)]
    
  # Start all queues for the eval graphs
  for m in eval_apps:
    with m['graph'].as_default():
      m['coord'] = tf.train.Coordinator()
      m['threads'] = tf.train.start_queue_runners(sess=m['sess'], coord=m['coord'])

  print('--------------------------------- TRAINING! ------------------------------------')
  try:
    # The step number
    step = 0

    # Init aggregators and names
    aggregators = {k: 0.0 for k in t_channels.keys()}
    # Operations that need to be executed in every training step
    step_ops = {'train_op': train_op}
    step_ops.update(t_channels)    
    summary_ops = {'train_op': train_op, 'summary_op': t_summary_op}
    summary_ops.update(t_channels)
    
    # Take time
    start = time.time()
    
    # Training loop
    proceed = True
    while step <= FLAGS.batches and not coord.should_stop() and proceed:  ###tag:Queues###
      # Save if needed
      if step % FLAGS.save_every == 0:
        if step != 0 or FLAGS.eval_beofre_training:
          print('Saving the model')
          t_saver.save(sess, os.path.join(CHECKPOINT_DIR, model_name), global_step=step, latest_filename=model_name)

      # Evaluate model is needed
      if step % FLAGS.eval_every == 0:
        if step != 0 or FLAGS.eval_beofre_training:
          print('Evaluating the model for step number {}'.format(step))
          current_test_channels = eval_once(eval_apps, 
                                            e_summary_writer, 
                                            model_name=model_name)
          if not best_test_channels:
            best_test_channels = current_test_channels
            best_step = step
          
          if action_manager:
            replace, proceed = action_manager(current_test_channels, 
                                              best_test_channels, 
                                              step)
            if replace:
              best_test_channels = current_test_channels
              best_step = step
        
      # Running the step
      if step % FLAGS.train_summary_every != 0:
        results = sess.run(step_ops)
        #print(sess.run(szs)) # For queues speed debugging
      
      else:
        results = sess.run(summary_ops)
        t_summary_writer.add_summary(results['summary_op'], 
                                     step)
        
      # Aggregate the train monitoring channels
      for key in aggregators.keys():
        aggregators[key] += results[key]
      
      # Monitor train channels if needed
      if step % FLAGS.train_monitor_every == 0:
        # Do some printing
        print('----- TRAIN MONITOR AFTER ANOTHER {} BATCHES ------------'.format(FLAGS.train_monitor_every))
        print('step number: {}'.format(step))
        for k in sorted(aggregators.keys()):
          v = aggregators[k]
          print('{}: {}'.format(k, 
                                v / float(FLAGS.train_monitor_every)))
        end = time.time(); print('seconds elapsed: {}'.format(end - start)) 
        
        start = end  # Measure time
        print('---------------------------------------------------------')
        print('')
        
        # Reset aggregators
        aggregators = {k: 0.0 for k in t_channels.keys()}

      # Increase the step
      step += 1
  except:
    raise
  finally:
    coord.request_stop()  ###tag:Queues###
    sess.close()
    coord.join(threads)  ###tag:Queues###
    for m in eval_apps:
      m['coord'].request_stop()
      m['sess'].close()
      m['coord'].join(m['threads'])
      
  return best_test_channels, best_step
