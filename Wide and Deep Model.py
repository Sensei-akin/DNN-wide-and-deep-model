import pandas as pd
from utils.utils import * 
import functools
import numpy as np
import tensorflow as tf
import yaml

def main():
    TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
    TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

    train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
    test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

    data = pd.read_csv(train_file_path)
    
    NUMERIC = ['age', 'n_siblings_spouses', 'parch', 'fare']

    numeric_columns = create_feature_cols(data, NUMERIC, use_normalization=False)

    CATEGORIES = {
        'sex': ['male', 'female'],
        'class' : ['First', 'Second', 'Third'],
        'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
        'embark_town' : ['Cherbourg', 'Southhampton', 'Queenstown'],
        'alone' : ['y', 'n']
    }

    categorical_columns = []
    embedding_categorical_columns = []
    for feature, vocab in CATEGORIES.items():
        cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
            key=feature, vocabulary_list=vocab)
        categorical_columns.append(tf.feature_column.indicator_column(cat_col))
        embedding_categorical_columns.append(tf.feature_column.embedding_column(cat_col, dimension=25))
        
    wide_features = categorical_columns + numeric_columns
    deep_features = categorical_columns + numeric_columns + embedding_categorical_columns

    model = model_fn(wide_features, deep_features)
    model.train(input_fn = lambda: get_dataset(train_file_path), steps=500)
    metrics = model.evaluate(input_fn= lambda: get_dataset(train_file_path, training =False), steps=1)

    feature_spec = tf.feature_column.make_parse_example_spec(wide_features + deep_features)

    # Build receiver function, and export.
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    export_dir = model.export_saved_model('exported_model', serving_input_receiver_fn)
    
if __name__ == "__main__":
    main()