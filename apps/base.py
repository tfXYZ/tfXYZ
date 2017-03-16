from __future__ import division, absolute_import, print_function
from abc import ABCMeta, abstractmethod
import tensorflow as tf, os


class BaseApp:
  """
  A base class for all apps.
  Apps that extend this class should always:
  - Set self.dataset_dir, self.tfr_structure_file, self.train_files, self.eval_files
  - Call super(BaseApp, self).__init__(param1, param2, ... , **kwargs)
  """
  __metaclass__ = ABCMeta
  
  def __init__(self, name, dataset_dir, train_files, eval_files):
    self.name = name
    self.dataset_dir = dataset_dir
    self.train_files = os.path.join(self.dataset_dir, train_files)
    self.eval_files = os.path.join(self.dataset_dir, eval_files)
    self.tfr_structure_file = os.path.join(self.dataset_dir, 'tfr_structure.json')
    
  def create_filename_queue(self, is_train, files, lengths):
    return tf.train.string_input_producer(files, capacity=500000)
      
  @abstractmethod
  def compute_loss(self, global_endpoints, module_endpoints):
    pass

  @abstractmethod
  def monitoring_channels(self, is_train, global_endpoints, module_endpoints):
    pass

  @abstractmethod
  def numpy_channels(self, concate_aggregators, step):
    pass

  @abstractmethod
  def precache_processing(self, is_train, val_dict):
    pass

  @abstractmethod
  def postcache_processing(self, is_train, val_dict, tfr_structure):
    pass
