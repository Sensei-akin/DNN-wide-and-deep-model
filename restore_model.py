# The following functions can be used to convert a value to a type compatible
# with tf.Example.
import tensorflow as tf
import numpy as np

TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)


import os
pwd = os.getcwd()
export_dire = pwd + '/exported_model'

file_list = []
def get_file_path(model_path):
    for root, dire, files in os.walk(model_path):
        file_list.append(root)
    export_dir = file_list[1]
    return export_dir

def get_dataset(file_path, training = True, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=1, # Artificially small to make examples easier to show.
      na_value="?",
      num_epochs=1,
      ignore_errors=True, 
      **kwargs)
    if training:
        dataset = dataset.shuffle(500).repeat()
    return dataset

def _bytes_feature(value):
  #"""Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  #"""Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  #""" Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Create a dictionary mapping the feature name to the tf.Example-compatible
# data type.
feature_dict = {
    'fare': _float_feature,
    'n_siblings_spouses': _float_feature,
    'parch': _float_feature,
    'age': _float_feature,
    'alone': _bytes_feature,
    'sex': _bytes_feature,
    'class': _bytes_feature,
    'deck': _bytes_feature,
    'embark_town': _bytes_feature,
}

def get_feature(ordered_dict):
    feature = {}
    for key,value in d.items():
        if key in feature_dict.keys():
            feature[key] = feature_dict[key](value.numpy()[0])
    return feature


def get_example(feature):
    return tf.train.Example(features=tf.train.Features(feature=feature))


export_dir = get_file_path(export_dire)
imported = tf.saved_model.load(export_dir)

def predict(example):
    return imported.signatures["predict"](
    examples=tf.constant([example.SerializeToString()]))



if __name__ == '__main__':
    eval_dataset = get_dataset(test_file_path, False, shuffle = False)
    
    new_list = []
    for d in eval_dataset.take(20):
        pred = predict(get_example(get_feature(d)))
        y = pred["probabilities"][0][1]
        new_list.append(y)
    predictions = np.array(new_list)

    import pandas as pd
    df = pd.read_csv(test_file_path).head(20)
    pred = pd.DataFrame(predictions, columns= ['investment_propensity'])
    new_df = pd.concat([df, pred], axis =1)
    new_df.to_csv('predictions.csv', index=False)
