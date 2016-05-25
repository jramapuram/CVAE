import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from cvae import CVAE

flags = tf.flags
flags.DEFINE_integer("latent_size", 20, "Number of latent variables.")
flags.DEFINE_integer("epochs", 100, "Maximum number of epochs.")
flags.DEFINE_integer("batch_size", 128, "Mini-batch size for data subsampling.")
flags.DEFINE_string("device", "/gpu:0", "Compute device.")
flags.DEFINE_boolean("allow_soft_placement", True, "Soft device placement.")
flags.DEFINE_float("device_percentage", "0.5", "Amount of memory to use on device.")
FLAGS = flags.FLAGS

def build_Nd_cvae(sess, source, input_shape, latent_size, batch_size, epochs=100):
    cvae = CVAE(sess, input_shape, batch_size, latent_size=latent_size)
    model_filename = "models/%s.cpkt" % cvae.get_name()
    if os.path.isfile(model_filename):
        cvae.load(sess, model_filename)
    else:
        sess.run(tf.initialize_all_variables())
        cvae.train(sess, source, batch_size, display_step=1, training_epochs=epochs)
        cvae.save(sess, model_filename)

    return cvae

# show clustering in 2d
def plot_2d_cvae(sess, source, cvae):
    x_sample, y_sample = source.test.next_batch(5000)
    z_mu = cvae.transform(sess, x_sample)
    plt.figure(figsize=(8, 6))
    plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
    plt.colorbar()
    plt.savefig("models/2d_cluster.png", bbox_inches='tight')
    #plt.show()

    # show reconstruction
def plot_Nd_cvae(sess, source, cvae, batch_size):
    x_sample = source.test.next_batch(batch_size)[0]
    x_reconstruct = cvae.reconstruct(sess, x_sample)
    plt.figure(figsize=(8, 12))
    for i in range(5):
        plt.subplot(5, 2, 2*i + 1)
        plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1)
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2*i + 2)
        plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1)
        plt.title("Reconstruction")
        plt.colorbar()
        plt.savefig("models/20d_reconstr_%d.png" % i, bbox_inches='tight')
        #plt.show()

def main():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    input_shape = int(np.sqrt(mnist.train.images.shape[1])), \
                  int(np.sqrt(mnist.train.images.shape[1]))

    # model storage
    if not os.path.exists('models'):
        os.makedirs('models')

    with tf.device(FLAGS.device):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.device_percentage)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                              gpu_options=gpu_options)) as sess:
            cvae = build_Nd_cvae(sess, mnist,
                                 input_shape,
                                 FLAGS.latent_size,
                                 FLAGS.batch_size,
                                 epochs=FLAGS.epochs)
            # 2d plot shows a cluster plot vs. a reconstruction plot
            if FLAGS.latent_size == 2:
                plot_2d_cvae(sess, mnist, cvae)
            else:
                plot_Nd_cvae(sess, mnist, cvae, FLAGS.batch_size)

if __name__ == "__main__":
    main()
