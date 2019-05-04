from helpers import lazy_property

import tensorflow as tf




class EnsembleNetwork(object):
    def __init__(
            self,
            ds_graph,
            num_neurons=[10, 5, 3],
            num_features=1,
            learning_rate=0.001,
            activations=None,  #[tf.nn.tanh,tf.nn.relu,tf.sigmoid]
            dropout_layers=None,  #[True,False,True]
            initialisation_scheme=None,  #[tf.random_normal,tf.random_normal,tf.random_normal]
            optimizer=None,  #defaults to GradiendDescentOptimizer,
            num_epochs=None,  #defaults to 1,
            seed=None,
            adversarial=None,
            initialisation_params=None,
            l2=None,
            l=None):

        #necessary parameters
        self.num_neurons = num_neurons
        self.ds_graph = ds_graph

        self.num_layers = len(num_neurons)
        self.num_features = num_features
        self.learning_rate = learning_rate or 0.001
        self.adversarial = adversarial or False
        self.initialisation_params = initialisation_params or {}
        self.l2 = l2 or False
        self.l = l or 0.05

        #optional parameters
        self.optimizer = optimizer or tf.train.AdamOptimizer  #tf.train.GradientDescentOptimizer
        self.activations = activations or [tf.nn.relu  #tf.nn.tanh
                                           ] * self.num_layers  #tanh,relu, 
        self.initialisation_scheme = initialisation_scheme or tf.keras.initializers.he_normal  ##tf.contrib.layers.xavier_initializer  #
        #tf.contrib.layers.xavier_initializer  #tf.truncated_normal  #tf.random_uniform#
        self.num_epochs = num_epochs or meta_num_epochs
        self.seed = seed or None

        self.initialise_graph
        self.initialise_session
        print('initialising Network {}'.format(type(self)))
        
    

    @lazy_property
    def initialise_graph(self):
        #initialise graph
        self.g = tf.Graph()
        #build graph with self.graph as default so nodes get appended
        with self.g.as_default():
            dataset_graph = tf.import_graph_def(self.ds_graph, return_elements = ['X:0',"y:0"])
            self.next = self.g.get_tensor_by_name('import/next:0')
            self.init_network
            self.predict_graph
            self.error_graph
            self.train_graph
            self.init = tf.global_variables_initializer()

    @lazy_property
    def initialise_session(self):
        #initialise session
        self.session = tf.Session(graph=self.g)
        #initialise global variables
        self.session.run(self.init)

    @lazy_property
    def init_network(self):
        if self.seed:
            tf.set_random_seed(self.seed)
        #inputs
        self.X = self.next[0]
        self.y = self.next[1]
        #lists for storage
        self.w_list = []
        self.b_list = []

        #add input x first weights
        #        self.w_list.append(
        #            tf.Variable(
        #                self.initialisation_scheme(
        #                    [self.num_features, self.num_neurons[0]]),
        #                name='w_0'))  #first Matrix
        initialiser = self.initialisation_scheme(seed=self.seed,
                                                 **self.initialisation_params)
        self.w_list.append(
            tf.Variable(
                initialiser([self.num_features, self.num_neurons[0]]),
                name='w_0'))

        #for each layer over 0 add a n x m matrix and a bias term
        for i, num_neuron in enumerate(self.num_neurons[1:]):
            n_inputs = self.num_neurons[i]  #for first hidden layer 3
            n_outputs = self.num_neurons[i + 1]  #for first hidden layer 5

            self.w_list.append(
                tf.Variable(
                    initialiser([n_inputs, n_outputs]), name='w_' + str(i)))
            self.b_list.append(
                tf.Variable(tf.ones(shape=[n_inputs]), name='b_' + str(i)))

        #add last layer m  x 1 for output
        self.w_list.append(
            tf.Variable(initialiser([self.num_neurons[-1], 1]),
                        name='w_-1'))  #this is a regression
        self.b_list.append(
            tf.Variable(
                tf.ones(shape=[self.num_neurons[-1]]), name='b_' + str(
                    len(self.num_neurons) + 1)))

    @lazy_property
    def predict_graph(self):
        #set layer_input to input
        layer_input = self.X

        #for each layer do
        for i, w in enumerate(self.w_list):

            #z = input x Weights
            a = tf.matmul(layer_input, w, name='matmul_' + str(i))

            #z + bias
            if i < self.num_layers:
            #if i > 0:
              bias = self.b_list[i]
              a = tf.add(a, bias)

            #a = sigma(z) if not last layer and regression
            if i < self.num_layers:

                a = self.activations[i](a)
            #set layer input to a for next cycle
            layer_input = a

        return a

    @lazy_property
    def error_graph(self):

        #y_hat is a // output of prediction graph
        y_hat = self.predict_graph

        #error is mean squared error of placehilder y and prediction
        error = tf.losses.mean_squared_error(self.y, y_hat)  #tf.square(
        #self.y - y_hat)  #

        if self.l2:
            # Loss function with L2 Regularization with beta=0.01
            regularizers = tf.reduce_sum(
                [tf.nn.l2_loss(weights) for weights in self.w_list])
            error = tf.reduce_mean(error + self.l * regularizers)

        return error

    @lazy_property
    def train_graph(self):

        #error is the error from error graph
        error = self.error_graph

        #optimizer is self.optimizer
        optimizer = self.optimizer(learning_rate=self.learning_rate)

        return optimizer.minimize(error)

    def train_offline(self, X, y):
        for epoch in range(self.num_epochs):

            self.session.run(self.train_graph,
                             feed_dict={self.X: X,
                                        self.y: y})

    def fit(self, X, y, shuffle=True, online=True):
        #print('X is {}'.format(X[:10]))

        #if shuffle:

        if online:
            for epoch in range(self.num_epochs):
                if shuffle:
                    X, y, _ = self.shuffle_data(X, y)
                X = self.check_input_dimensions(X)
                y = self.check_input_dimensions(y)
                self.train_one_epoch(X, y, shuffle)
        else:
            if shuffle:
                X, y, _ = self.shuffle_data(X, y)
            X = self.check_input_dimensions(X)
            y = self.check_input_dimensions(y)
            self.train_offline(X, y)
            
            
    def fit(self,epochs):
      for epoch in range(epochs):
        self.session.run(self.train_graph)
        

    def train_one_epoch(self, epoch_X, epoch_y, shuffle=True):

        if shuffle == True:
            epoch_X, epoch_y, _ = self.shuffle_data(
                #np.squeeze(epoch_X), np.squeeze(epoch_y))
                epoch_X,
                epoch_y)
        #print(epoch_X[:10])
        epoch_X = self.check_input_dimensions(epoch_X)
        epoch_y = self.check_input_dimensions(epoch_y)
        for batch_X, batch_y in zip(epoch_X, epoch_y):
            self.train_one(batch_X, batch_y)
            if self.adversarial:
                self.train_one(batch_X + 0.05, batch_y)
                self.train_one(batch_X - 0.05, batch_y)

    def train_one(self, batch_X, batch_y):
        batch_X = self.check_input_dimensions(batch_X)
        batch_y = self.check_input_dimensions(batch_y)
        batch_X = np.array(batch_X).T

        batch_X, _, batch_y, _ = train_test_split(
            batch_X, batch_y, test_size=0.00, random_state=self.seed)

        #print('what')
        #print('X size: {}'.format(batch_X.shape))
        #print('y size: {}'.format(batch_y.shape))
        self.session.run(self.train_graph,
                         feed_dict={self.X: batch_X,
                                    self.y: batch_y})

    def train_and_evaluate(self, X, y, shuffle=False):
        errors = []
        for epoch in range(self.num_epochs):

            #self.session.run(self.train_graph,
            #                 feed_dict={self.X: X,
            #                            self.y: y})
            self.train_one_epoch(X, y, shuffle)

            errors += list(np.sqrt((y - self.predict(X))**2))

            #self.session.run(self.predict_graph,
            #                          feed_dict={self.X: X,
            #                                     self.y: y}))**2))

        return errors

    def predict(self, X):
        X = self.check_input_dimensions(X)

        return self.session.run(self.predict_graph,
                                feed_dict={self.X: X}).squeeze()

    def kill(self):
        self.session.close()

    def check_input_dimensions(self, array):
        """Makes sure arrays are compatible with Tensorflow input
        can't have array.shape = (X,),
        needs to be array.shape = (X,1)"""
        #y = array
        #y = np.reshape(y, [y.shape[0], 1])
        #return y
        if len(array.shape) <= 1:

            return np.expand_dims(array, 1)
        else:

            return array

    def shuffle_data(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        sorted_index = np.argsort(p)
        p = np.squeeze(p)
        return a[p], b[p], sorted_index

    def network_mutli_dimensional_scatterplot(self, X_test, y_test, X=None,
                                              y=None, figsize=(20, 50),
                                              filename=None):

        #y_hat = self.predict(X_test)
        #print(pred_dict)
        #std = self.predict_var(X_test)
        y_hat, std = self.get_prediction_and_std(X_test)

        #plt.rcParams["figure.figsize"] = (20,20)
        fig = plt.figure(figsize=figsize)
        #plt.scatter(X[:,5],y)

        num_features = len(X_test.T)
        for i, feature in enumerate(X_test.T):
            #sort the arrays
            s = np.argsort(feature)
            var = y_hat[s] - std[s]
            var2 = y_hat[s] + std[s]
            print(feature.shape)
            print(var.shape)

            plt.subplot(num_features, 1, i + 1)
            plt.plot(
                feature[s],
                y_hat[s],
                label='predictive Mean', )
            plt.fill_between(feature[s].ravel(), y_hat[s].ravel(),
                             var.ravel(), alpha=.3, color='b',
                             label='uncertainty')
            plt.fill_between(feature[s].ravel(), y_hat[s].ravel(),
                             var2.ravel(), alpha=.3, color='b')
            plt.scatter(feature[s], y_test[s], label='data', s=20,
                        edgecolor="black", c="darkorange")
            plt.xlabel("data")
            plt.ylabel("target")
            plt.title("Ensemble")
            plt.legend()
            if filename is not None:
                plt.savefig(filename)

        if filename is not None:
            plt.savefig(filename)
        #plt.show()
        return fig

    def compute_rsme(self, X, y):
        y_hat = self.predict(X)
        #y_hat = pred_dict['means']
        #std = pred_dict['stds']        #print(y_hat.shape,std.shape,y.shape)

        return np.sqrt(np.mean((y_hat - y)**2))

    def score(self, X, y):
        return self.compute_rsme(X, y)

    def compute_error_vec(self, X, y):
        y_hat = self.predict(X)
        return y - y_hat