from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import numpy as np
import tensorflow as tf

## provides and example of how to load CSV data from a file into a tf.data.Dataset

## Data taken from the Titanic passenger list. 
# The model will predict the likelihood a passenger survived based on characteristics like age, gender ticket class, and wheter the person was traveling alone.


TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

## explicitly identify the column with the value the model is intended to predict
LABEL_COLUMN = 'survived'
LABELS = [0, 1]

## Read the CSV data from the file and create a dataset
def get_dataset(file_path, **kwargs):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=5,  # Artificially small to make examples easier to show.
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True,
      **kwargs)
  return dataset


raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)


def show_batch(dataset):
  for batch, label in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key, value.numpy()))


print("\n")
show_batch(raw_train_data)
print("\n")
SELECT_COLUMNS = ['survived', 'age',
                  'n_siblings_spouses', 'class', 'deck', 'alone']

temp_dataset = get_dataset(train_file_path, select_columns=SELECT_COLUMNS)

show_batch(temp_dataset)
