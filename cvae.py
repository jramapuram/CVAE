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
    def __init__(self, sess, input_shape, batch_size, latent_size=128, e_dim=64, d_dim=64):
        self.input_shape = input_shape
        self.input_size = np.prod(input_shape)
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.e_dim = e_dim
        self.d_dim = d_dim
        self.iteration = 0

        self.inputs = tf.placeholder(tf.float32, [None, self.input_size], name="inputs")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

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

        self.loss, self.optimizer = self._create_loss_and_optimizer(self.inputs,
                                                                    self.x_reconstr_mean,
                                                                    self.z_log_sigma_sq,
                                                                    self.z_mean)
        self.loss_summary = tf.scalar_summary("loss", self.loss)
        self.summaries = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter("./logs", sess.graph)
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
        #     the data and some prior. This acts as a kind of regularize.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        self.latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq
                                           - tf.square(z_mean)
                                           - tf.exp(z_log_sigma_sq), 1)
        loss = tf.reduce_mean(self.reconstr_loss + self.latent_loss)   # average over batch

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
        return loss, optimizer

    def _encode_step(self, x, outfilters, name, filter_x=5, filter_y=5):
        def _get_conv_train():
            return lrelu(batch_norm(conv2d(x, outfilters, name=name+"_econv",
                                           k_h=filter_y, k_w=filter_x),
                                    self.is_training, name=name+"_ebn"),
                         name=name+"e_h")

        def _get_conv_test():
            return lrelu(batch_norm(conv2d(x, outfilters, name=name+"_econv",
                                           k_h=filter_y, k_w=filter_x,
                                           reuse=True), self.is_training,
                                    name=name+"_ebn", reuse=True),
                         name=name+'e_h', reuse=True)

        return tf.cond(self.is_training, _get_conv_train, _get_conv_test)

    def _get_cond_linear(self, x, out_dim, name, activ):
        def _get_train():
            return activ(linear(x, out_dim, scope=name+"_lin"),
                         name=name)
        def _get_test():
            return activ(linear(x, out_dim, scope=name+"_lin", reuse=True),
                      name=name)

        return tf.cond(self.is_training, _get_train, _get_test)

    def encoder(self, inputs, latent_size, activ=tf.identity):
        with tf.variable_scope("encoder"):
            i = tf.reshape(inputs, [-1,
                                    self.input_shape[0],
                                    self.input_shape[1],
                                    1], name="e_i")
            def _e0_train():
                return lrelu(conv2d(i, self.e_dim, name="e_conv0"), name="e_h0_conv")
            def _e0_test():
                return lrelu(conv2d(i, self.e_dim, name="e_conv0", reuse=True),
                             name="e_h0_conv", reuse=True)
            e0 = tf.cond(self.is_training, _e0_train, _e0_test)

            e1 = self._encode_step(e0, self.e_dim*2, name="e_conv1")
            e2 = self._encode_step(e1, self.e_dim*4, name="e_conv2")

            result_size = self.e_dim * 4 * 16
            unrolled = tf.reshape(e2, [-1, result_size], name="e_unrolled")
            z_mean = self._get_cond_linear(unrolled, latent_size, "z_mean", activ)
            z_sigma_sq = self._get_cond_linear(unrolled, latent_size, "z_sigma_sq", activ)

            return z_mean, z_sigma_sq

    def _decode_step(self, x, filter_x, filter_y, outfilters, name):
        def _get_deconv_train():
            return tf.nn.relu(batch_norm(deconv2d(x, [self.batch_size,
                                                      filter_x, filter_y,
                                                      outfilters],
                                                  name=name+"_dconv"),
                                         self.is_training, name=name+"_dbn"),
                              name=name+'d_h')

        def _get_deconv_test():
            return tf.nn.relu(batch_norm(deconv2d(x, [self.batch_size,
                                                      filter_x, filter_y,
                                                      outfilters],
                                                  name=name+"_dconv", reuse=True),
                                         self.is_training, name=name+"_dbn", reuse=True),
                              name=name+'d_h')

        return tf.cond(self.is_training, _get_deconv_train, _get_deconv_test)

    def decoder(self, z, projection_size, activ=tf.identity):
        with tf.variable_scope("decoder"):
            z_ = self._get_cond_linear(z, self.d_dim * 8 * 4 * 4,
                                       name="d_h0_lin", activ=tf.identity)

            def _d0_train():
                return tf.nn.relu(batch_norm(tf.reshape(z_, [-1, 4, 4, self.d_dim * 8]),
                                             self.is_training), name="d0")

            def _d0_test():
                return tf.nn.relu(batch_norm(tf.reshape(z_, [-1, 4, 4, self.d_dim * 8]),
                                             self.is_training, reuse=True),
                                  name="d0")

            d0 = tf.cond(self.is_training, _d0_train, _d0_test)

            d1 = self._decode_step(d0, 8, 8, self.d_dim*4, "d_conv1")
            d2 = self._decode_step(d1, 16, 16, self.d_dim*2, "d_conv2")
            d3 = self._decode_step(d2, 32, 32, self.d_dim, "d_conv3")

            def _d4_train():
                return tf.nn.relu(deconv2d(d3, [self.batch_size, 64, 64, 1],
                                           name="dconv4"),
                                  name='d_h4')

            def _d4_test():
                return tf.nn.relu(deconv2d(d3, [self.batch_size, 64, 64, 1],
                                           name="dconv4", reuse=True),
                                  name='d_h4')

            d4 = tf.cond(self.is_training, _d4_train, _d4_test)

            result_size = self.d_dim * 4 * 16
            unrolled = tf.reshape(d4, [-1, result_size], name="d_unrolled")
            x_reconstr_mean = self._get_cond_linear(unrolled, projection_size,
                                                    name="z_mean_decoder", activ=activ)
            return x_reconstr_mean

    def partial_fit(self, sess, X):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        feed_dict = {self.inputs: X,
                     self.is_training: True}

        if self.iteration % 10 == 0:
            _, summary, cost  = sess.run([self.optimizer, self.summaries, self.loss],
                                         feed_dict=feed_dict)
            self.summary_writer.add_summary(summary, self.iteration)
        else:
            _, cost  = sess.run([self.optimizer, self.loss],
                                feed_dict=feed_dict)

        self.iteration += 1
        return cost

    def transform(self, sess, inputs):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        feed_dict={self.inputs: inputs,
                   self.is_training: False}
        return sess.run(self.z_mean,
                        feed_dict=feed_dict)

    def generate(self, sess):
        """ Generate data by sampling from latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        feed_dict={self.z: z_mu,
                   self.is_training: False}
        return sess.run(self.x_reconstr_mean,
                        feed_dict=feed_dict)

    def reconstruct(self, sess, X):
        """ Use VAE to reconstruct given data. """
        feed_dict={self.inputs: X,
                   self.is_training: False}
        return sess.run(self.x_reconstr_mean,
                        feed_dict=feed_dict)

    def train(self, sess, source, batch_size, training_epochs=1, display_step=5):
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

    def init_all(self, sess):
        sess.run(tf.initialize_all_variables(), feed_dict={self.is_training: True})
        sess.run(tf.initialize_all_variables(), feed_dict={self.is_training: False})



######## entry point ########
def main():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    input_shape = int(np.sqrt(mnist.train.images.shape[1])), \
                  int(np.sqrt(mnist.train.images.shape[1]))
    batch_size = 128

    with tf.device("/cpu:0"):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            cvae = CVAE(sess, input_shape, batch_size, latent_size=2)
            cvae.init_all(sess)
            cvae.train(sess, mnist, batch_size, display_step=1, training_epochs=1)

            x_sample, y_sample = mnist.test.next_batch(5000)
            cvae.batch_size = 5000
            z_mu = cvae.transform(sess, x_sample)
            plt.figure(figsize=(8, 6))
            plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
            plt.colorbar()
            plt.show()




if __name__ == "__main__":
    main()
######## /entry point ########
