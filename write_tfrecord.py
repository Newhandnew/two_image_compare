import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import cv2

flags = tf.app.flags
flags.DEFINE_string("image_folder", "defectCmpLabelData", "image folder with compare images")
flags.DEFINE_string("tfrecord_name", "2_image_compare", "name of tfrecord")
flags.DEFINE_string("label_list_path", "defectCmpLabelData/defect_cmp_label.txt", "path of label list")

FLAGS = flags.FLAGS


def transfer_tfrecord(image_array, label):
    tfrecord_feature = {'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}
    # refactor in the future
    for index, image in enumerate(image_array):
        bytes_image = image.tobytes()
        tfrecord_feature['img_{}'.format(index)] = \
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_image]))
    tf_transfer = tf.train.Example(features=tf.train.Features(feature=tfrecord_feature))
    return tf_transfer


def write_image_list(image_names, file_name):
    with open(file_name, 'w+') as f:
        for image in image_names:
            f.write('{} '.format(image))


def read_image_array(pattern_path_list, image_folder):
    image_array = []
    for path in pattern_path_list:
        image_path = os.path.join(image_folder, path)
        image = cv2.imread(image_path, 0)
        image_array.append(image)
    return image_array


if __name__ == "__main__":
    assert FLAGS.image_folder, "--picture_folder necessary"
    assert FLAGS.tfrecord_name, "--tfrecord_name necessary"
    assert FLAGS.label_list_path, "--label_list_path necessary"
    label_list = []
    with open(FLAGS.label_list_path) as f:
        for line in f:
            label_list.append(line.strip().split())

    print(label_list)
    data_number = len(label_list) * 2
    print('data number: {}'.format(data_number))

    output_dir = 'data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tfrecord_train = FLAGS.tfrecord_name + '_train.tfrecords'
    tfrecord_test = FLAGS.tfrecord_name + '_test.tfrecords'
    output_train = os.path.join(output_dir, tfrecord_train)
    output_test = os.path.join(output_dir, tfrecord_test)
    test_ratio = 0.2
    writer_train = tf.python_io.TFRecordWriter(output_train)
    writer_test = tf.python_io.TFRecordWriter(output_test)
    total_train_size = 0
    total_test_size = 0
    train_file_path = os.path.join(output_dir, FLAGS.tfrecord_name + '_train_list')
    test_file_path = os.path.join(output_dir, FLAGS.tfrecord_name + '_test_list')

    for label in range(2):
        train_image, test_image = train_test_split(label_list, test_size=test_ratio, random_state=123)
        print('process label {} training data...'.format(label))
        for image_pair in train_image:
            if label == 1:
                image_pair.reverse()
            print(image_pair)
            image_array = read_image_array(image_pair, FLAGS.image_folder)
            tf_transfer = transfer_tfrecord(image_array, label)
            writer_train.write(tf_transfer.SerializeToString())
        print('process label {} testing data...'.format(label))
        for image_pair in test_image:
            if label == 1:
                image_pair.reverse()
            print(image_pair)
            image_array = read_image_array(image_pair, FLAGS.image_folder)
            tf_transfer = transfer_tfrecord(image_array, label)
            writer_test.write(tf_transfer.SerializeToString())
        total_train_size += len(train_image)
        total_test_size += len(test_image)

    writer_train.close()
    writer_test.close()

    print('done! train size: {}, test size: {}'.format(total_train_size, total_test_size))



