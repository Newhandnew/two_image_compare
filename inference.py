import tensorflow as tf
import os
import time
import alexnet
import cv2

slim = tf.contrib.slim

flags = tf.app.flags
flags.DEFINE_string("image_folder", "defectCmpLabelData", "image folder with compare images")
flags.DEFINE_string('logs_dir', 'compare',
                    'Directory to save the checkpoints and training summaries.')
flags.DEFINE_string("label_list_path", "defectCmpLabelData/defect_cmp_label.txt", "path of label list")
FLAGS = flags.FLAGS


def read_image_array(pattern_path_list, image_folder):
    image_array = []
    for path in pattern_path_list:
        image_path = os.path.join(image_folder, path)
        image = cv2.imread(image_path, 0)
        image_array.append(image)
    return image_array


def main(_):
    """
    Configuration Part.
    """
    assert FLAGS.image_folder, "--picture_folder necessary"
    assert FLAGS.logs_dir, "--logs_dir necessary"
    assert FLAGS.label_list_path, "--label_list_path necessary"
    logs_path = os.path.join('logs', FLAGS.logs_dir)
    pattern_extension = range(2)
    num_classes = 2
    label_list = []
    with open(FLAGS.label_list_path) as f:
        for line in f:
            label_list.append(line.strip().split())

    image_size = [256, 256]

    image1_placeholder = tf.placeholder(tf.uint8, [None, image_size[0], image_size[1]], name='image1_input')
    image2_placeholder = tf.placeholder(tf.uint8, [None, image_size[0], image_size[1]], name='image2_input')

    merged_image = tf.stack((image1_placeholder, image2_placeholder), -1)
    float_input_tensor = tf.to_float(merged_image)
    # Define the network
    # with slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
    #     logits, _ = mobilenet_v2.mobilenet(tf.to_float(image_tensor), num_classes=num_classes)
    with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
        logits, end_points = alexnet.alexnet_v2(float_input_tensor, num_classes=num_classes, is_training=False)

    predictions = tf.argmax(logits, 1, name='output_argmax')
    # Setup the global step.
    tf.train.get_or_create_global_step()
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    tf.logging.set_verbosity(tf.logging.INFO)
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session(config=session_config) as sess:
        start_time = time.time()
        prev_model = tf.train.get_checkpoint_state(logs_path)
        if prev_model:
            saver.restore(sess, prev_model.model_checkpoint_path)
            elapsed_time = time.time() - start_time
            print('Checkpoint found, {}'.format(prev_model))
            print('restore elapsed time: {}'.format(elapsed_time))
            for image_pair in label_list:
                image_array = read_image_array(image_pair, FLAGS.image_folder)
                start_time = time.time()
                predict_array = sess.run(predictions, feed_dict={image1_placeholder: [image_array[0]],
                                                                 image2_placeholder: [image_array[1]]})
                elapsed_time = time.time() - start_time
                print("Prediction: {}, shape: {}".format(predict_array, predict_array.shape))
                print('inference elapsed time: {}'.format(elapsed_time))

        else:
            print('No checkpoint found')


if __name__ == '__main__':
    tf.app.run()