""" Adapted from https://github.com/deepmind/multi_object_datasets.
"""

from os.path import exists
from os.path import join
from os import makedirs
import tensorflow as tf
import numpy as np
import argparse


def main(tfrecord_path, data_path):
    train_thresh = 60000
    test_tresh = 320

    ds = dataset(join(tfrecord_path, "tetrominoes_train.tfrecords"))

    # create directories
    train_img_dir = join(data_path, "train/images/")
    train_mask_dir = join(data_path, "train/masks/")
    test_img_dir = join(data_path, "test/images/")
    test_mask_dir = join(data_path, "test/masks/")
    if not exists(data_path):
        makedirs(data_path)
    if not exists(train_img_dir):
        makedirs(train_img_dir)
    if not exists(train_mask_dir):
        makedirs(train_mask_dir)
    if not exists(test_img_dir):
        makedirs(test_img_dir)
    if not exists(test_mask_dir):
        makedirs(test_mask_dir)

    shapes_train, colors_train = [], []
    shapes_test, colors_test = [], []
    for i, sample in enumerate(ds):
        if i < train_thresh:
            np.save(join(train_img_dir, "image_" + str(i)), sample["image"].numpy())
            np.save(join(train_mask_dir, "mask_" + str(i)), sample["mask"].numpy())
            shapes_train.append(sample["shape"].numpy())
            colors_train.append(sample["color"].numpy())
        elif i >= train_thresh and i < train_thresh + test_tresh:
            np.save(join(test_img_dir, "image_"+str(i)), sample["image"].numpy())
            np.save(join(test_mask_dir, "mask_"+str(i)), sample["mask"].numpy())
            shapes_test.append(sample["shape"].numpy())
            colors_test.append(sample["color"].numpy())
        else:
            break
    np.save(join(data_path, "train/shapes"), np.array(shapes_train))
    np.save(join(data_path, "train/colors"), np.array(colors_train))
    np.save(join(data_path, "test/shapes"), np.array(shapes_test))
    np.save(join(data_path, "test/colors"), np.array(colors_test))


def _decode(example_proto):
    IMAGE_SIZE = [35, 35]
    MAX_NUM_ENTITIES = 4
    BYTE_FEATURES = ['mask', 'image']

    features = {
        'image': tf.io.FixedLenFeature(IMAGE_SIZE+[3], tf.string),
        'mask': tf.io.FixedLenFeature([MAX_NUM_ENTITIES]+IMAGE_SIZE+[1], tf.string),
        'x': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
        'y': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
        'shape': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
        'color': tf.io.FixedLenFeature([MAX_NUM_ENTITIES, 3], tf.float32),
        'visibility': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
        }
    # Parse the input `tf.Example` proto using the feature description dict above.
    single_example = tf.io.parse_single_example(example_proto, features)
    for k in BYTE_FEATURES:
        single_example[k] = tf.squeeze(tf.io.decode_raw(single_example[k], tf.uint8),
                                    axis=-1)
    return single_example


def dataset(tfrecords_path, read_buffer_size=None, map_parallel_calls=None):
    """ Read, decompress, and parse the TFRecords file.

        Args:
            tfrecords_path: str. Path to the dataset file.
            read_buffer_size: int. Number of bytes in the read buffer. See documentation
            for `tf.data.TFRecordDataset.__init__`.
            map_parallel_calls: int. Number of elements decoded asynchronously in
            parallel. See documentation for `tf.data.Dataset.map`.

        Returns:
            An unbatched `tf.data.TFRecordDataset`.
    """
    raw_dataset = tf.data.TFRecordDataset(
        tfrecords_path, compression_type=tf.io.TFRecordOptions.get_compression_type_string('GZIP'),
        buffer_size=read_buffer_size)
    return raw_dataset.map(_decode, num_parallel_calls=map_parallel_calls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord_path", default=None, type=str, help="Path where the tfrecord is located", required=True)
    parser.add_argument("--data_path", default=None, type=str, help="Path where the dataset should be stored", required=True)
    args = parser.parse_args()
    args = vars(args)
    main(args["tfrecord_path"], args["data_path"])