#!/usr/local/bin/python
import sys
import os
import argparse
from glob import glob
from statistics import mean
import logging
import tensorflow as tf
from datasets import preprocess, restore_image, car_labels
from schedulers import WarmupScheduler

import smdebug.tensorflow as smd
from smdebug.core.collection import CollectionKeys
from smdebug.tensorflow import SaveConfig

def create_dataset(file_pattern, batch_size=32, train=True):
    tdf = tf.data.TFRecordDataset(glob(file_pattern))
    tdf = tdf.map(preprocess)
    if train:
        tdf = tdf.shuffle(batch_size*8)
    tdf = tdf.batch(batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)
    tdf = iter(tdf)
    return tdf

#s@tf.function
def train_step(images, labels, model, optimizer, loss_func):
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss = loss_func(labels, logits)
        scaled_loss = optimizer.get_scaled_loss(loss)
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    #gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

#s@tf.function
def predict(images, model):
    logits = model(images, training=False)
    probs = tf.math.softmax(logits)
    pred = tf.math.top_k(probs)
    return pred

#s@tf.function
def eval_step(images, labels, model):
    probs, preds = predict(images, model)
    accuracy = tf.reduce_mean(tf.cast(tf.squeeze(preds) == labels, tf.float32))
    return accuracy

if __name__=='__main__':
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
    tf.config.optimizer.set_jit(True)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--steps_per_epoch', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=2e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    
    args, _ = parser.parse_known_args()
    
    
    model = tf.keras.applications.ResNet50(include_top=True, weights=None, classes=196)
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(name='loss')
    # decay at 50 and 75% of training
    lr_decay_steps = [int(args.steps_per_epoch * args.num_epochs * .5), int(args.steps_per_epoch * args.num_epochs * .75)]
    lr_decay_levels = [args.learning_rate, args.learning_rate/10, args.learning_rate/100]
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(lr_decay_steps, lr_decay_levels)
    lr_schedule = WarmupScheduler(lr_schedule, args.learning_rate/10, args.warmup_steps)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=args.momentum)
    optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, 128)

    # create datasets
    data_dir = os.environ['SM_CHANNEL_ALL_DATA']

    train_file_pattern = os.path.join(data_dir, 'cars196-train*')
    eval_file_pattern = os.path.join(data_dir, 'cars196-test*')
    
    train_tdf = create_dataset(train_file_pattern, args.batch_size, train=True)
    eval_tdf = create_dataset(eval_file_pattern, args.batch_size, train=False)
    
    loss_history = []
    accuracy_history = []
    for epoch in range(args.num_epochs):
        print("starting epoch {}".format(epoch + 1))
        for step in range(args.steps_per_epoch):
            images, labels = next(train_tdf)
            loss_history.append(train_step(images, labels, model, optimizer, loss_func).numpy())
            current_learning_rate = lr_schedule(optimizer.iterations)
            if step%100==0:
                print("training:step {0}, training:loss {1:.4f}, training:lr {2:.4f}".format(step,
                                                                            mean(loss_history[-50:]),
                                                                            lr_schedule(optimizer.iterations).numpy()))
        for step in range(args.eval_steps):
            images, labels = next(eval_tdf)
            accuracy_history.append(eval_step(images, labels, model).numpy())
        print("eval:accuracy {0:.4f}".format(mean(accuracy_history[-args.eval_steps:])))
