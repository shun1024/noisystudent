from absl import flags


def experimental():
  # Experiment configs
  flags.DEFINE_string(
    'model_dir', default=None,
    help=('The directory where the model and training/evaluation summaries are'
          ' stored.'))

  flags.DEFINE_bool(
    'init_model', default=False,
    help='whether to initialize the student')

  flags.DEFINE_string(
    'init_model_path', default=None,
    help='initialize the student from checkpoint')

  flags.DEFINE_string(
    'model_name',
    default='efficientnet-b0',
    help='The model name among existing configurations.')

  flags.DEFINE_string(
    'mode', default='train_and_eval',
    help='One of {train_and_eval, train, eval}.')

  flags.DEFINE_integer(
    'input_image_size', default=None,
    help='Input image size: it depends on specific model name.')

  # Cloud TPU Cluster Resolvers
  flags.DEFINE_bool(
    'use_tpu', default=True,
    help=('Use TPU to execute the model for training and evaluation. If'
          ' --use_tpu=false, will use whatever devices are available to'
          ' TensorFlow by default (e.g. CPU and GPU)'))

  flags.DEFINE_string(
    'master', default=None,
    help='not used')

  flags.DEFINE_string(
    'tpu', default=None,
    help='The Cloud TPU to use for training. This should be either the name '
         'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

  flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
         'will attempt to automatically detect the GCE project from metadata.')

  flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
         'will attempt to automatically detect the GCE project from metadata.')

  flags.DEFINE_integer('debug', 0, '')

  flags.DEFINE_integer('log_step_count_steps', 64, 'The number of steps at '
                                                   'which the global step information is logged.')
  flags.DEFINE_bool(
    'use_cache', default=True, help='Enable cache for training input.')

  flags.DEFINE_integer(
    'keep_checkpoint_max', default=5, help='Number of checkpoints to keep.')

  flags.DEFINE_integer(
    'num_train_shards', default=None, help='Number of training shards to use.')

  flags.DEFINE_bool(
    'small_image_model', default=False, help='whether the image size is 32x32')

  flags.DEFINE_integer(
    'num_tpu_cores', default=None, help='not used')

  flags.DEFINE_integer(
    'save_checkpoints_steps', default=1000,
    help='Batch size for training.')


def data_info():
  # Data Specification
  flags.DEFINE_string(
    'task_name', default='imagenet', help='imagenet or svhn')

  flags.DEFINE_string(
    'label_data_dir',
    default=None,
    help='The directory where the labeled data is stored.')

  flags.DEFINE_integer(
    'num_train_images', default=None, help='Size of training data set.')

  flags.DEFINE_integer(
    'num_eval_images', default=None, help='Size of validation data set.')

  flags.DEFINE_integer(
    'num_test_images', default=None, help='Size of test data set.')

  flags.DEFINE_integer(
    'steps_per_eval', default=3000,
    help=('Controls how often evaluation is performed. Since evaluation is'
          ' fairly expensive, it is advised to evaluate as infrequently as'
          ' possible (i.e. up to --train_steps, which evaluates the model only'
          ' after finishing the entire training regime).'))

  flags.DEFINE_bool(
    'skip_host_call', default=False,
    help=('Skip the host_call which is executed every training step. This is'
          ' generally used for generating training summaries (train loss,'
          ' learning rate, etc...). When --skip_host_call=false, there could'
          ' be a performance drop if host_call function is slow and cannot'
          ' keep up with the TPU-side computation.'))

  flags.DEFINE_integer(
    'iterations_per_loop', default=1000,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

  flags.DEFINE_string(
    'data_format', default='channels_last',
    help=('A flag to override the data format used in the model. The value'
          ' is either channels_first or channels_last. To run the network on'
          ' CPU or TPU, channels_last should be used. For GPU, channels_first'
          ' will improve performance.'))

  flags.DEFINE_integer(
    'num_label_classes', default=1000, help='Number of classes, at least 2')

  flags.DEFINE_bool(
    'transpose_input', default=True,
    help='Use TPU double transpose optimization')

  flags.DEFINE_bool(
    'use_bfloat16',
    default=True,
    help='Whether to use bfloat16 as activation for training.')


def basic_learning():
  # Basic Learning Parameter
  flags.DEFINE_integer(
    'train_steps', default=109474,
    help='The number of steps to use for training. 350 epochs on ImageNet.')

  flags.DEFINE_float(
    'train_ratio', default=1.0,
    help=('The train_steps and decay steps are multiplied by train_ratio.'
          'When train_ratio > 1, training is going to take longer.'))

  flags.DEFINE_integer(
    'train_batch_size', default=4096, help='Batch size for training.')

  flags.DEFINE_integer(
    'eval_batch_size', default=8, help='Batch size for evaluation.')

  flags.DEFINE_float(
    'base_learning_rate',
    default=0.016,
    help='Base learning rate when train batch size is 256.')

  flags.DEFINE_float(
    'moving_average_decay', default=0.9999,
    help='Moving average decay rate.')

  flags.DEFINE_float(
    'weight_decay', default=1e-5,
    help='Weight decay coefficiant for l2 regularization.')

  flags.DEFINE_float(
    'label_smoothing', default=0.1,
    help='Label smoothing parameter used in the softmax_cross_entropy')

  flags.DEFINE_float(
    'dropout_rate', default=None,
    help='Dropout rate for the final output layer.')

  flags.DEFINE_float(
    'stochastic_depth_rate', default=None,
    help='Drop connect rate for the network.')

  flags.DEFINE_float(
    'depth_coefficient', default=None,
    help='Depth coefficient for scaling number of layers.')

  flags.DEFINE_float(
    'width_coefficient', default=None,
    help='Width coefficient for scaling channel size.')

  flags.DEFINE_float(
    'final_base_lr', default=None, help='final learning rate.')


def noisy_student():
  # Noisy Student Parameter
  flags.DEFINE_string(
    'unlabel_data_dir', default='', help='unlabeled data dir')

  flags.DEFINE_float(
    'unlabel_ratio', default=0,
    help='batch size of unlabeled data: unlabel_ratio * train_batch_size')

  flags.DEFINE_float(
    'teacher_softmax_temp', default=-1,
    help=('The softmax temperature when teacher computes the predicted distribution.'
          '-1 means to use an one-hot distribution'))

  flags.DEFINE_integer(
    'train_last_step_num', -1,
    ('Used for fine tuning. Only train for train_last_step_num out of the '
     'total train_steps'))

  flags.DEFINE_string(
    'teacher_model_name', default=None,
    help='the model_name of the teacher model')

  flags.DEFINE_string(
    'teacher_model_path', default=None,
    help='teacher model checkpoint path')

  flags.DEFINE_string(
    'augment_name', default=None,
    help='None: normal cropping and flipping. v1: RandAugment')

  flags.DEFINE_bool(
    'remove_aug', False,
    help='Whether to use center crop for augmentation')

  flags.DEFINE_integer(
    'fix_layer_num', default=-1,
    help='Fix the first fix_layer_num layers when fine tuning')

  flags.DEFINE_integer(
    'randaug_mag', default=27, help='randaugment magnitude')

  flags.DEFINE_integer(
    'randaug_layer', default=2, help='number of ops in randaugment')

  flags.DEFINE_integer(
    'num_shards_per_group', default=-1,
    help='Tpu specific batch norm hyperparameters')

  flags.DEFINE_float(
    'label_data_sample_prob', default=1,
    help=('Tpu specific hyperparameter. On Tpu, there should be at least one '
          'labeled image on each core. When we want to use a train_batch_size '
          'smaller than the num_tpu_cores, we set this hyperparameter to mask '
          'out some labeled images in the loss function. '))

  flags.DEFINE_string(
    'unl_aug', 'default', 'augmentation for unlabeled data.')

  flags.DEFINE_bool(
    'cutout_op', default=True, help='use cutout in RandAugment')


def define_flags():
  experimental()
  data_info()
  basic_learning()
  noisy_student()
