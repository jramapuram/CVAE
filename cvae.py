import tensorflow as tf
import numpy as np

from utils import *

batch_size = 128

class CVAE(object):
    '''
    CVAE: Convolutional Variational AutoEncoder

    Builds a convolutional variational autoencoder that compresses
    input_size to latent_size and then back out again.

    max_downsampled: the threshold of conv downsampling [after this we FC]
    max_upsampled: the threshold of conv upsampling [after this we FC]
    '''

    def __init__(self, input_shape, batch_size, latent_size=128, num_filter_init=64,
                 filter_size=[5,5], strides=[2, 2],
                 max_downsampled=6000, max_upsampled=None):
        self.input_shape = input_shape
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.num_filter_init = 64
        self.filter_size = filter_size
        self.strides = strides
        # self.max_downsampled = int(max_downsampled) if max_downsampled else int(self.latent_size * 2)
        self.max_upsampled = int(max_upsampled) if max_upsampled else int(np.prod(self.input_shape) * 0.75)
        self.max_downsampled = max_downsampled
        self.is_training = False

        if len(input_shape) == 2:
            self.inputs = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1]])
        elif len(input_shape) == 1:
            self.inputs = tf.placeholder(tf.float32, [None, input_shape[0]])
        else:
            raise Exception("invalid input shape")

        # self.z = tf.placeholder(tf.float32, [None, latent_size])
        # self.z_summary = tf.histogram_summary("z", self.z)

        # Encode our data into z and return the mean and covariance
        self.z_mean, self.z_log_sigma_sq = self.encoder(self.inputs, filter_size
                                                        , num_filter_init, strides, latent_size,
                                                        self.max_downsampled)

        # z = mu + sigma * epsilon
        # epsilon is a sample from a N(0, 1) distribution
        eps = tf.random_normal([batch_size, latent_size], 0.0, 1.0, dtype=tf.float32)
        self.z = tf.add(self.z_mean,
                        tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        self.z_summary = tf.histogram_summary("z", self.z)

        # Get the reconstructed mean from the decoder
        self.x_reconstr_mean = self.decoder(self.z, filter_size,
                                            strides, latent_size,
                                            self.max_downsampled)

        self.loss, self.optimizer = self._create_loss_and_optimizer(self.inputs,
                                                                    self.x_reconstr_mean,
                                                                    self.z_log_sigma_sq,
                                                                    self.z_mean)
        self.loss_summary = tf.scalar_summary("loss", self.loss)

    def _create_loss_and_optimizer(self, inputs, x_reconstr_mean, z_log_sigma_sq, z_mean):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluation of log(0.0)
        reconstr_loss = \
            -tf.reduce_sum(inputs * tf.log(1e-10 + x_reconstr_mean)
                           + (1 - inputs) * tf.log(1e-10 + 1 - x_reconstr_mean),
                           1)
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        ##    between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq
                                           - tf.square(z_mean)
                                           - tf.exp(z_log_sigma_sq), 1)
        loss = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
        return loss, optimizer

    def encoder(self, inputs, latent_size, activ=tf.identity):
        e0 = lrelu(conv2d(inputs, self.e_dim, name="e_h0_conv"))

        bn1 = batch_norm(self.batch_size, name="e1_bn")
        e1 = lrelu(bn1(conv2d(e0, self.e_dim*2, name="e_h1_conv")))

        bn2 = batch_norm(self.batch_size, name="e2_bn")
        e2 = lrelu(bn1(conv2d(e0, self.e_dim*4, name="e_h2_conv")))

        bn3 = batch_norm(self.batch_size, name="e3_bn")
        e3 = lrelu(bn1(conv2d(e0, self.e_dim*8, name="e_h3_conv")))

        unrolled = tf.reshape(e3, [-1])
        z_mean = activ(linear(unrolled, latent_size, "z_mean"))
        z_sigma_sq = activ(linear(unrolled, latent_size, "z_sigma_sq"))
        return z_mean, z_sigma_sq

    def decoder(self, z, projection_size, activ=tf.identity):
        z_ = linear(z, self.d_dim*8*4*4, 'd_h0_lin')
        bn0 = batch_norm(self.batch_size, name="d0_bn")
        d0 = tf.nn.relu(bn0(tf.reshape(z_, [-1, 4, 4, self.d_dim * 8])))

        bn1 = batch_norm(self.batch_size, name="d1_bn")
        d1 = tf.nn.relu(bn1(deconv2d(d0, [self.batch_size,
                                          8, 8,
                                          self.d_dim*4],
                                     name='d_h1')))

        bn2 = batch_norm(self.batch_size, name="d2_bn")
        d2 = tf.nn.relu(bn2(deconv2d(d0, [self.batch_size,
                                          16, 16,
                                          self.d_dim*2],
                                     name='d_h2')))

        bn3 = batch_norm(self.batch_size, name="d3_bn")
        d3 = tf.nn.relu(bn3(deconv2d(d0, [self.batch_size,
                                          32, 32,
                                          self.d_dim*1],
                                     name='d_h3')))

        d4 = tf.nn.relu(deconv2d(d0, [self.batch_size,
                                      64, 64, 1],
                                 name='d_h4'))

        unrolled = tf.reshape(d4, [-1])
        x_reconstr_mean = activ(linear(unrolled,
                                       projection_size,
                                       "z_mean_decoder"))
        return x_reconstr_mean


    # def decoder(self, inputs, filter_size, num_filter_init, strides, latent_size, max_upsampled):
    #     ifilters = num_filter_init; ofilters = 1
    #     first_proj_size = num_filter_init * 8 * 4* 4
    #     idims = [self.batch_size, first_proj_size]

    #     # first upsample so that we can start convolving
    #     d0_linear = linear(inputs, first_proj_size, 'decoder_linear')
    #     d0_unrolled = tf.reshape(d0_linear, [-1, 4, 4, num_filter_init*8])
    #     bn = batch_norm(batch_size, name=str(idims).replace("[", "dbn_").replace("]", "").replace(", ", "_"))
    #     d0 = tf.nn.relu(bn(d0_unrolled))
    #     deconv = [d0]

    #     print 'convolutional upsampling params:'
    #     while idims[0] >= filter_size[0]:
    #         print '%s filter = ' % filter_size, [filter_size[0], filter_size[1],
    #                                              ifilters, ofilters], ' | input_size = ', idims,
    #         idims[0] *= 2
    #         idims[1] *= 2
    #         bn = batch_norm(batch_size, name=str(idims).replace("[", "dbn_").replace("]", "").replace(", ", "_"))
    #         deconv.append(lrelu(bn(deconv2d(inputs, ofilters, k_h=filter_size[0], k_w=filter_size[1],
    #                                d_h=filter_size[0], d_w=filter_size[1])),
    #                                name=str(idims).replace("[", "c_").replace("]", "").replace(", ", "_")))
    #         ifilters = ofilters
    #         ofilters = num_filter_init * 2


    # def encoder(self, inputs, filter_size, num_filter_init, strides, latent_size, max_downsampled):
    #     ifilters = 1; ofilters = num_filter_init
    #     if len(self.input_shape) == 1 :
    #         idims = [self.batch_size, self.input_shape[0]]
    #     else:
    #         idims = [self.batch_size, self.input_shape[0], self.input_shape[1]]

    #     inputs_unrolled = tf.reshape(inputs, [-1, 1, 1, self.input_size])

    #     print 'convolutional downsampling params:'
    #     bn = batch_norm(self.batch_size, name=str(idims).replace("[", "ebn_").replace("]", "").replace(", ", "_"))
    #     c0 = lrelu(bn(conv2d(inputs_unrolled, ofilters, k_h=filter_size[0], k_w=filter_size[1],
    #                       d_h=strides[0], d_w=strides[1])),
    #                name=str(idims).replace("[", "c_").replace("]", "").replace(", ", "_"))
    #     print '%s filter = ' % filter_size, [filter_size[0], filter_size[1],
    #                                          ifilters, ofilters], ' | input_size = ', idims,
    #     idims[0] = self._div_round(idims[0], [strides[0]])
    #     idims[1] = self._div_round(idims[1], [strides[1]])
    #     conv = [c0]

    #     # Add conv ops until we reach the threshold of being <= max_downsampled
    #     while idims[0] >= filter_size[0] and idims[1] >= filter_size[1] and np.prod(idims) >= max_downsampled:
    #         bn = batch_norm(self.batch_size, name=str(idims).replace("[", "ebn_").replace("]", "").replace(", ", "_"))
    #         conv.append(lrelu(bn(conv2d(conv[-1], ofilters, k_h=filter_size[0], k_w=filter_size[1],
    #                                  d_h=filter_size[0], d_w=filter_size[1])),
    #                           name=str(idims).replace("[", "c_").replace("]", "").replace(", ", "_")))
    #         print '%s filter = ' % filter_size, [filter_size[0], filter_size[1],
    #                                              ifilters, ofilters], ' | input_size = ', idims,
    #         ifilters = ofilters
    #         ofilters = num_filter_init * 2
    #         # if idims[0] >= 2 and ofilters < 16:
    #         #     ofilters = ofilters * 2
    #         # elif idims[0] >= 2:
    #         #     ofilters = ofilters / 2

    #     unrolled = tf.reshape(conv[-1], [-1])
    #     z_mean = linear(unrolled, latent_size, "z_mean")
    #     z_sigma_sq = linear(unrolled, latent_size, "z_sigma_sq")
    #     return z_mean, z_sigma_sq


    # def build_encoder_params(self, input_size, latent_size, max_downsampled):
    #     W = []; b = []; dstype = []; filter_list = []; bn = []
    #     ifilters = 1; ofilters = 32
    #     idims = deepcopy(input_size)
    #     print 'convolutional downsampling params:'
    #     while idims[0] >= 2:
    #         dstype.append(['2x2'])
    #         init = tf.contrib.layers.xavier_initializer_conv2d()
    #         W.append(tf.Variable(init(shape=[2, 2, ifilters, ofilters])))
    #         b.append(tf.Variable(tf.constant(0.01, shape=[ofilters])))
    #         bn.append(batch_norm(self.batch_size, name='d_bn_%s' % str(idims)))
    #         print '[2x2] filter = ', [2, 2, ifilters, ofilters], ' | input_size = ', idims,
    #         filter_list.append(ofilters)
    #         idims[1] = self._div_round(idims[1], [2])
    #         idims[0] = self._div_round(idims[0], [2])
    #         ifilters = ofilters
    #         if idims[0] >= 2 and ofilters < 16:
    #             ofilters = ofilters * 2
    #         elif idims[0] >= 2:
    #             ofilters = ofilters / 2

    #         print ' | output_dims = ', idims

    #     while idims[1] >= max_downsampled:
    #         dstype.append(['1x1', '1x2'])
    #         init = tf.contrib.layers.xavier_initializer_conv2d()
    #         W.append(tf.Variable(tf.random_uniform(shape=[1, 2, ifilters, ofilters], minval=-0.05, maxval=0.05)))
    #         b.append(tf.Variable(tf.constant(0.01, shape=[ofilters])))
    #         bn.append(batch_norm(self.batch_size, name='d_bn_%s' % str(idims)))
    #         print '[1x2] filter = ', [1, 2, ifilters, ofilters], ' | input_size = ', idims,
    #         filter_list.append(ofilters)
    #         idims[1] = self._div_round(idims[1], [2])
    #         ifilters = ofilters
    #         ofilters = ofilters / 2 if ofilters > 2 and idims[1] >= max_downsampled else ofilters
    #         print ' | output_dims = ', idims

    #     # add the FC layers
    #     total_params = ofilters * idims[1] * idims[0]
    #     W.append(tf.Variable(xavier_init(total_params, latent_size), trainable=True))
    #     b.append(tf.Variable(tf.zeros([latent_size]), trainable=True))

    #     return W, b, bn, dstype, filter_list

    # def build_decoder_params(self, input_size, latent_size, max_upsampled):
    #     W = []; b = []; dstype = []; filter_list = []; bn = []
    #     ifilters = 1; ofilters = 32
    #     idims = deepcopy(input_size)
    #     print 'convolutional upsampling params:'

    #     while idims[1] <= max_upsampled:
    #         dstype.append(['1x1', '1x2'])
    #         init = tf.contrib.layers.xavier_initializer_conv2d()
    #         W.append(tf.Variable(tf.random_uniform(shape=[1, 2, ifilters, ofilters], minval=-0.05, maxval=0.05)))
    #         b.append(tf.Variable(tf.constant(0.01, shape=[ofilters])))
    #         print '[1x2] filter = ', [1, 2, ifilters, ofilters], ' | input_size = ', idims,
    #         filter_list.append(ofilters)
    #         idims[1] = self._div_round(idims[1], [2])
    #         ifilters = ofilters
    #         ofilters = ofilters / 2 if ofilters > 2 and idims[1] >= latent_size else ofilters
    #         print ' | output_dims = ', idims

    #     # add the FC layers
    #     total_params = ofilters * idims[1] * idims[0]
    #     W.append(tf.Variable(xavier_init(total_params, latent_size), trainable=True))
    #     b.append(tf.Variable(tf.zeros([latent_size]), trainable=True))

    #     return W, b, dstype, filter_list

    # # run the operation for a single element
    # def _do_conv(self, inputs, W, b, bn, ctype, filter_size, index):
    #     if ctype[0] == '2x2' and ctype[1] == '2x2':
    #         conv = lrelu(bn(conv2d(inputs, self.df_dim*2, name='d_h1_conv')))
    #         conv = conv_relu_2x2(inputs, W, b, filter_size, self.is_training, index)
    #         #pool = max_pool_2x2(conv)
    #     elif ctype[0] == '2x2' and ctype[1] == '1x2':
    #         conv = conv_relu_2x2(inputs, W, b, filter_size, self.is_training, index)
    #         #pool = max_pool_1x2(conv)
    #     else:
    #         conv = conv_relu_1x1(inputs, W, b, filter_size, self.is_training, index)
    #         #pool = max_pool_1x2(conv)

    #     return conv#, pool

    # def conv_fc(self, inputs, idims, W, b, bn, dstype, filter_list):
    #     # handle 0th case because of inputs
    #     index = 0
    #     # c0, p0 = self._do_conv(tf.reshape(inputs, [-1, idims[0], idims[1], 1])
    #     #                        , W[0], b[0], dstype[0], filter_list[0], 'bn' + str(index))
    #     c0 = self._do_conv(tf.reshape(inputs, [-1, idims[0], idims[1], 1])
    #                        , W[0], b[0], bn[0], dstype[0], filter_list[0], 'bn' + str(index))

    #     conv = [c0]#; pool = [p0]
    #     for weight, bias, conv_type, filter_dim in zip(W[1:-1], b[1:-1], dstype[1:], filter_list[1:]):
    #         index += 1
    #         #c, p = self._do_conv(pool[-1], weight, bias, conv_type, filter_dim, 'bn' + str(index))
    #         c = self._do_conv(conv[-1], weight, bias, conv_type, filter_dim, 'bn' + str(index))
    #         conv.append(c)
    #         #pool.append(p)

    #     # do the final dense downsample
    #     #pool_flat = tf.reshape(pool[-1], [-1])
    #     #return tf.nn.relu(tf.matmul(tf.expand_dims(pool_flat, dim=0), W[-1]) + b[-1])
    #     flat = tf.reshape(conv[-1], [-1])
    #     return tf.nn.relu(tf.matmul(tf.expand_dims(flat, dim=0), W[-1]) + b[-1])

    def partial_fit(self, X):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.loss),
                                  feed_dict={self.x: X})
        return cost


    def transform(self, inputs):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.inputs: inputs})

    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.z: z_mu})

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.x: X})

    def train(self, source, batch_size=100, training_epochs=10, display_step=5):
        n_samples = source.train.num_examples
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_samples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, _ = source.train.next_batch(batch_size)

                # Fit training using batch data
                cost = vae.partial_fit(batch_xs)
                # Compute average loss
                avg_cost += cost / n_samples * batch_size

            # Display logs per epoch step
            if epoch % display_step == 0:
                print "Epoch:", '%04d' % (epoch+1), \
                      "cost=", "{:.9f}".format(avg_cost)
        return vae


######## entry point ########
def main():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    input_shape = np.sqrt(mnist.train.images.shape[1]), \
                  np.sqrt(mnist.train.images.shape[1])

    cvae = CVAE(input_shape, batch_size, latent_size=2)




if __name__ == "__main__":
    main()
######## /entry point ########
