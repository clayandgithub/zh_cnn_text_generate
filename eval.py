#! /usr/bin/env python
# encoding: UTF-8

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import word2vec_helpers
from word2vec_map_cnn import Word2VecMapCNN
import csv

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("seed_text", '', "The seed text for generating new text.")
tf.flags.DEFINE_integer("num_word_to_generate", 1280, "The number of word to be generated.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size (default: 1)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# validate
# ==================================================

# discover the max_checkpoint_dir if checkpoint_dir is empty
checkpoint_dir = FLAGS.checkpoint_dir
if checkpoint_dir == "":
    max_checkpoint_dir = "0"
    runs_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
    for dir_path in os.listdir(runs_dir):
        if os.path.isdir(os.path.join(runs_dir, dir_path)) and dir_path > max_checkpoint_dir:
            max_checkpoint_dir = dir_path
    checkpoint_dir = os.path.join(runs_dir, max_checkpoint_dir, "checkpoints")

# validate checkout point file
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
if checkpoint_file is None:
    print("Cannot find a valid checkpoint file!")
    exit(0)
print("Using checkpoint file : {}".format(checkpoint_file))

# validate word2vec model file
trained_word2vec_model_file = os.path.join(checkpoint_dir, "..", "trained_word2vec.model")
if not os.path.exists(trained_word2vec_model_file):
    print("Word2vec model file \'{}\' doesn't exist!".format(trained_word2vec_model_file))
print("Using word2vec model file : {}".format(trained_word2vec_model_file))

# validate training params file
training_params_file = os.path.join(checkpoint_dir, "..", "training_params.pickle")
if not os.path.exists(training_params_file):
    print("Training params file \'{}\' is missing!".format(training_params_file))
print("Using training params file : {}".format(training_params_file))

# Load params
params = data_helpers.loadDict(training_params_file)
window_size = int(params['window_size'])
print("params = {}".format(params))

# Load data
generated_text = []
# seed_text = FLAGS.seed_text.strip()
seed_text = u'白 玉 京'
x_text_current = [] if len(seed_text) == 0 else seed_text.split(' ')
generated_text.extend(x_text_current)
x_text_current = [data_helpers.sentence_start_padding(x_text_current, window_size)]

# Get Embedding vector x_test
x_current = np.array(word2vec_helpers.embedding_sentences(x_text_current, file_to_load = trained_word2vec_model_file))
print("x_current.shape = {}".format(x_current.shape))
print("x_current = {}".format(x_current))

# Generation
# ==================================================
print("\nGenerating...\n")
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        #batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        w2vModel = word2vec_helpers.load_model(trained_word2vec_model_file)
        for word_count in range(FLAGS.num_word_to_generate):
            current_predictions = sess.run(predictions, {input_x: x_current, dropout_keep_prob: 1.0})
            current_word = word2vec_helpers.sample_similar_word_by_vector(w2vModel, current_predictions[0])
            generated_text.append(current_word)
            if (current_word == '<END_PADDING>'):
                generated_text.append('\n')
                x_current = [[w2vModel.wv['<START_PADDING>']] * window_size]
            else:
                x_current = np.array([np.concatenate((x_current[0][1:], current_predictions), axis = 0)])
            # all_predictions = np.concatenate([all_predictions, batch_predictions])

# Save the evaluation to a csv
out_path = os.path.join(checkpoint_dir, "..", "prediction.csv")
print("Saving generated text to {0}".format(out_path))
with open(out_path, 'w') as f:
    for word in generated_text:
        f.write(word.encode('utf-8'))
