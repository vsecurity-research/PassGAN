# Libraries
import os
import sys
import time
import pickle
import argparse
import tensorflow as tf
import numpy as np

# Files
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import utils
import models

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB * 2 of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 2)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs", file=sys.stderr)
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

'''
python sample.py \
	--input-dir pretrained \
	--checkpoint pretrained/checkpoints/checkpoint_200000.ckpt \
	--output generated_pass.txt \
	--batch-size 1024 \
	--num-samples 1000000
'''


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-dir', '-i',
                        required=True,
                        dest='input_dir',
                        help='Trained model directory. The --output-dir value used for training.')

    parser.add_argument('--checkpoint', '-c',
                        required=True,
                        dest='checkpoint',
                        help='Model checkpoint to use for sampling. Expects a .ckpt file.')

    parser.add_argument('--output', '-o',
                        default='samples.txt',
                        help='File path to save generated samples to (default: samples.txt)')

    parser.add_argument('--num-samples', '-n',
                        type=int,
                        default=1000000,
                        dest='num_samples',
                        help='The number of password samples to generate (default: 1000000)')

    parser.add_argument('--batch-size', '-b',
                        type=int,
                        default=64,
                        dest='batch_size',
                        help='Batch size (default: 64).')
    
    parser.add_argument('--seq-length', '-l',
                        type=int,
                        default=10,
                        dest='seq_length',
                        help='The maximum password length. Use the same value that you did for training. (default: 10)')
    
    parser.add_argument('--layer-dim', '-d',
                        type=int,
                        default=128,
                        dest='layer_dim',
                        help='The hidden layer dimensionality for the generator. Use the same value that you did for training (default: 128)')

    parser.add_argument('--stdout',
                        help='Whether or not to output results to stdout indefinitely',
                        action='store_true')
    
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        parser.error('"{}" folder doesn\'t exist'.format(args.input_dir))

    if not os.path.exists(args.checkpoint + '.meta'):
        parser.error('"{}.meta" file doesn\'t exist'.format(args.checkpoint))

    if not os.path.exists(os.path.join(args.input_dir, 'charmap.pickle')):
        parser.error('charmap.pickle doesn\'t exist in {}, are you sure that directory is a trained model directory'.format(args.input_dir))

    if not os.path.exists(os.path.join(args.input_dir, 'charmap_inv.pickle')):
        parser.error('charmap_inv.pickle doesn\'t exist in {}, are you sure that directory is a trained model directory'.format(args.input_dir))

    return args

args = parse_args()

# Dictionary
with open(os.path.join(args.input_dir, 'charmap.pickle'), 'rb') as f:
    charmap = pickle.load(f, encoding='latin1')

# Reverse-Dictionary
with open(os.path.join(args.input_dir, 'charmap_inv.pickle'), 'rb') as f:
    inv_charmap = pickle.load(f, encoding='latin1')
    

tf.compat.v1.disable_eager_execution()

fake_inputs = models.Generator(args.batch_size, args.seq_length, args.layer_dim, len(charmap))

with tf.compat.v1.Session() as session:

    def generate_samples():
        samples = session.run(fake_inputs)
        samples = np.argmax(samples, axis=2)
        decoded_samples = []
        for i in range(len(samples)):
            decoded = []
            for j in range(len(samples[i])):
                decoded.append(inv_charmap[samples[i][j]])
            decoded_samples.append(tuple(decoded))
        return decoded_samples

    def save(samples):
        with open(args.output, 'a') as f:
                for s in samples:
                    s = "".join(s).replace('`', '')
                    f.write(s + "\n")

    def output(samples):
        for s in samples:
            s = "".join(s).replace('`','')
            print(s)

    saver = tf.compat.v1.train.Saver()
    saver.restore(session, args.checkpoint)

    samples = []
    then = time.time()
    start = time.time()

    if args.stdout:
        i = 0
        while True:
            samples.extend(generate_samples())

            # append to output file every 100 batches
            if i % 100 == 0 and i > 0: 
                
                output(samples)
                samples = [] # flush

                #print('wrote {} samples to {} in {:.2f} seconds. {} total.'.format(100 * args.batch_size, args.output, time.time() - then, i * args.batch_size), file=sys.stderr)
                then = time.time()
                i = 0
            else:
                i += 1
        
        output(samples)
    else:
        for i in range(int(args.num_samples / args.batch_size)):        
            samples.extend(generate_samples())

            # append to output file every 1000 batches
            if i % 1000 == 0 and i > 0: 
                
                save(samples)
                samples = [] # flush

                print('wrote {} samples to {} in {:.2f} seconds. {} total.'.format(1000 * args.batch_size, args.output, time.time() - then, i * args.batch_size), file=sys.stderr)
                then = time.time()
        
        save(samples)

print('\nFinished in {:.2f} seconds'.format(time.time() - start), file=sys.stderr)