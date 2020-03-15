import functools
        
import numpy as np
import tensorflow as tf

LABEL_COLUMN = 'survived'
def get_dataset(file_path, training = True, **kwargs):
    '''Returns a tf dataset'''
    dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=200,
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True, 
      **kwargs)
    if training:
        dataset = dataset.shuffle(500).repeat()
    return dataset


def get_normalization_parameters(data, features):
    """Get the normalization parameters (E.g., mean, std) for traindf for 
    features. We will use these parameters for training, eval, and serving."""

    def _z_score_params(column):
        
        mean = data[column].mean()
        std = data[column].std()
        return {'mean': mean, 'std': std}

    normalization_parameters = {}
    for column in features:
        normalization_parameters[column] = _z_score_params(column)
    return normalization_parameters


def numeric_column_normalized(column_name, normalizer_fn):
    return tf.feature_column.numeric_column(column_name, normalizer_fn=normalizer_fn)

def make_zscaler(mean, std):
    def zscaler(col):
        return (col - mean)/std
    return zscaler

# Define your feature columns
def create_feature_cols(data, features, use_normalization):
    """Create feature columns using tf.feature_column. 
    
    This function will get executed during training, evaluation, and serving."""
    
    normalization_parameters = get_normalization_parameters(data, features)
    
    def normalize_column(col):  # Use mean, std defined below.
        return (col - mean)/std
    normalized_feature_columns = []
    for column_name in features:
        normalizer_fn = None
        if use_normalization:
            column_params = normalization_parameters[column_name]
            mean = column_params['mean']
            std = column_params['std']
            normalizer_fn = make_zscaler(mean, std)
        normalized_feature_columns.append(numeric_column_normalized(column_name,
                                                                     normalizer_fn))
    return normalized_feature_columns


def model_fn(wide_features, deep_features):
    model = tf.estimator.DNNLinearCombinedClassifier(
        model_dir='titanic-checkpoints',
        linear_feature_columns=(wide_features),
        dnn_feature_columns=(deep_features),
        dnn_hidden_units=[250, 250],
        dnn_activation_fn=tf.nn.relu,
        batch_norm=True,
        dnn_optimizer = tf.optimizers.Adam(learning_rate = 0.004))
    return model