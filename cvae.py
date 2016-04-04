import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils import *

class CVAE(object):
    '''
    CVAE: Convolutional Variational AutoEncoder

    Builds a convolutional variational autoencoder that compresses
    input_shape to latent_size and then back out again. It uses
    the reparameterization trick and conv/conv transpose to achieve this.

    '''
    def __init__(self, input_shape, batch_size, latent_size=128, e_dim=64, d_dim=64):
        self.input_shape = input_shape
        self.input_size = np.prod(input_shape)
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.e_dim = e_dim
        self.d_dim = d_dim
        self.is_training = False

        self.inputs = tf.placeholder(tf.float32, [None, self.input_size])

        # Encode our data into z and return the mean and covariance
        self.z_mean, self.z_log_sigma_sq = self.encoder(self.inputs, latent_size)

        # z = mu + sigma * epsilon
        # epsilon is a sample from a N(0, 1) distribution
        eps = tf.random_normal([batch_size, latent_size], 0.0, 1.0, dtype=tf.float32)
        self.z = tf.add(self.z_mean,
                        tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        self.z_summary = tf.histogram_summary("z", self.z)

        # Get the reconstructed mean from the decoder
        self.x_reconstr_mean = self.decoder(self.z, self.input_size)
        self.x_reconstr_mean_sampler = self.sampler(self.z, self.input_size)

        self.loss, self.optimizer = self._create_loss_and_optimizer(self.inputs,
                                                                    self.x_reconstr_mean,
                                                                    self.z_log_sigma_sq,
                                                                    self.z_mean)
        self.loss_summary = tf.scalar_summary("loss", self.loss)
        self.saver = tf.train.Saver()

    def _create_loss_and_optimizer(self, inputs, x_reconstr_mean, z_log_sigma_sq, z_mean):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluation of log(0.0)
        self.reconstr_loss = \
            -tf.reduce_sum(inputs * tf.log(tf.clip_by_value(x_reconstr_mean, 1e-10, 1.0))
                           + (1 - inputs) * tf.log(tf.clip_by_value(1 - x_reconstr_mean, 1e-10, 1.0)),
                           1)
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        ##    between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        self.latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq
                                           - tf.square(z_mean)
                                           - tf.exp(z_log_sigma_sq), 1)
        loss = tf.reduce_mean(self.reconstr_loss + self.latent_loss)   # average over batch

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
        return loss, optimizer

    def encoder(self, inputs, latent_size, activ=tf.identity):
        i = tf.reshape(inputs, [self.batch_size,
                                self.input_shape[0],
                                self.input_shape[1], 1])
        e0 = lrelu(conv2d(i, self.e_dim, name="e_h0_conv"))

        bn1 = batch_norm(self.batch_size, name="e1_bn")
        e1 = lrelu(bn1(conv2d(e0, self.e_dim*2, name="e_h1_conv")))

        bn2 = batch_norm(self.batch_size, name="e2_bn")
        e2 = lrelu(bn2(conv2d(e1, self.e_dim*4, name="e_h2_conv")))

        # bn3 = batch_norm(self.batch_size, name="e3_bn")
        # e3 = lrelu(bn3(conv2d(e2, self.e_dim*8, name="e_h3_conv")))

        unrolled = tf.reshape(e2, [self.batch_size, -1])
        z_mean = activ(linear(unrolled, latent_size, scope="z_mean"))
        z_sigma_sq = activ(linear(unrolled, latent_size, scope="z_sigma_sq"))
        return z_mean, z_sigma_sq

    def decoder(self, z, projection_size, activ=tf.identity):
        z_ = linear(z, self.d_dim*8*4*4, scope='d_h0_lin')
        bn0 = batch_norm(self.batch_size, name="d0_bn")
        d0 = tf.nn.relu(bn0(tf.reshape(z_, [-1, 4, 4, self.d_dim * 8])))

        bn1 = batch_norm(self.batch_size, name="d1_bn")
        d1 = tf.nn.relu(bn1(deconv2d(d0, [self.batch_size,
                                          8, 8,
                                          self.d_dim*4],
                                     name='d_h1')))

        bn2 = batch_norm(self.batch_size, name="d2_bn")
        d2 = tf.nn.relu(bn2(deconv2d(d1, [self.batch_size,
                                          16, 16,
                                          self.d_dim*2],
                                     name='d_h2')))

        bn3 = batch_norm(self.batch_size, name="d3_bn")
        d3 = tf.nn.relu(bn3(deconv2d(d2, [self.batch_size,
                                          32, 32,
                                          self.d_dim*1],
                                     name='d_h3')))

        d4 = tf.nn.relu(deconv2d(d3, [self.batch_size,
                                      64, 64, 1],
                                 name='d_h4'))

        unrolled = tf.reshape(d4, [self.batch_size, -1])
        x_reconstr_mean = activ(linear(unrolled,
                                       projection_size,
                                       scope="z_mean_decoder"))
        return x_reconstr_mean

    def sampler(self, z, projection_size, activ=tf.identity):
        tf.get_variable_scope().reuse_variables()

        z_ = linear(z, self.d_dim*8*4*4, scope='d_h0_lin')
        bn0 = batch_norm(self.batch_size, name="d0_bn")
        d0 = tf.nn.relu(bn0(tf.reshape(z_, [-1, 4, 4, self.d_dim * 8])))

        bn1 = batch_norm(self.batch_size, name="d1_bn")
        d1 = tf.nn.relu(bn1(deconv2d(d0, [self.batch_size,
                                          8, 8,
                                          self.d_dim*4],
                                     name='d_h1')))

        bn2 = batch_norm(self.batch_size, name="d2_bn")
        d2 = tf.nn.relu(bn2(deconv2d(d1, [self.batch_size,
                                          16, 16,
                                          self.d_dim*2],
                                     name='d_h2')))

        bn3 = batch_norm(self.batch_size, name="d3_bn")
        d3 = tf.nn.relu(bn3(deconv2d(d2, [self.batch_size,
                                          32, 32,
                                          self.d_dim*1],
                                     name='d_h3')))

        d4 = tf.nn.relu(deconv2d(d3, [self.batch_size,
                                      64, 64, 1],
                                 name='d_h4'))

        unrolled = tf.reshape(d4, [self.batch_size, -1])
        x_reconstr_mean = activ(linear(unrolled,
                                       projection_size,
                                       scope="z_mean_decoder"))
        return x_reconstr_mean

    def partial_fit(self, sess, X):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        _, cost  = sess.run([self.optimizer, self.loss],
                            feed_dict={self.inputs: X})
        return cost


    def transform(self, sess, inputs):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return sess.run(self.z_mean,
                        feed_dict={self.inputs: inputs})

    def generate(self, sess, z_mu=None):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return sess.run(self.x_reconstr_mean_sampler,
                        feed_dict={self.z: z_mu})

    def reconstruct(self, sess, X):
        """ Use VAE to reconstruct given data. """
        return sess.run(self.x_reconstr_mean_sampler,
                        feed_dict={self.x: X})

    def train(self, sess, source, batch_size, training_epochs=1, display_step=5):
        # summaries = tf.merge_all_summaries()
        # summary_writer = tf.train.SummaryWriter("./logs", sess.graph)

        n_samples = source.train.num_examples
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_samples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, _ = source.train.next_batch(batch_size)

                # Fit training using batch data
                cost = self.partial_fit(sess, batch_xs)
                # Compute average loss
                avg_cost += cost / n_samples * batch_size

            # Display logs per epoch step
            #if epoch % display_step == 0:
                print "[Epoch:", '%04d]' % (epoch+1), \
                    "current cost = ", "{:.9f} | ".format(cost), \
                    "avg cost = ", "{:.9f}".format(avg_cost)


######## entry point ########
def main():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    input_shape = int(np.sqrt(mnist.train.images.shape[1])), \
                  int(np.sqrt(mnist.train.images.shape[1]))
    batch_size = 128

    with tf.device("/gpu:0"):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            cvae = CVAE(input_shape, batch_size, latent_size=2)
            sess.run(tf.initialize_all_variables())
            cvae.train(sess, mnist, batch_size, display_step=1)

            x_sample, y_sample = mnist.test.next_batch(5000)
            z_mu = cvae.transform(sess, x_sample)
            plt.figure(figsize=(8, 6))
            plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
            plt.colorbar()
            plt.show()




if __name__ == "__main__":
    main()
######## /entry point ########
