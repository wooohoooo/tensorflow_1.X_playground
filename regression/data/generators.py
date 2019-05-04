import numpy as np
import tensorflow as tf

class ToyDataset():
    """Toy Dataset generator that can provide numpy arrays, TF datasets or the graph of a tf dataset"""
    def __init__(self,datalen,shuffle):
        self.shuffle = shuffle
        self.datalen = datalen
        self.X, self.y = self.generate_data()
        self.tf_dataset = self.generate_tf_dataset()
        self.iterator = self.generate_tf_dataset_iterator()
        
    def graph(self):
        """ensure that each time we take a graph its newly created. 
        Otherwise we run into some tensorflow issues that I am too lazy to check out right now"""
        return self.generate_tf_dataset_iterator_graph()
    
    def generate_data_alternative(self) -> np.array:
        """legacy, ignore"""


        X = np.linspace(0,50,datalen)

        noise = [np.random.random() * 0.3 for i in range(self.datalen)]


        y = np.sin(X*0.2+10)+2 + X//30 +noise

        X = (X - np.min(X)) / (np.max(X) - np.min(X))
        X, y = np.expand_dims(X.astype(np.float32),1), np.expand_dims(y.astype(np.float32),1)


        plt.plot(X,y)




    def generate_data(self) -> np.array:
        """returns numpy arrays X and y that can be used as basis for regression problem"""
        
        X = np.arange(0, self.datalen)
        freq1 = 0.2
        freq2 = 0.15

        freq1 = 0.1
        freq2 = 0.0375
        noise = [np.random.random() * 0.1 for i in range(self.datalen)]
        y1 = np.sin(X * freq1) + noise
        y2 = np.sin(X * freq2) + noise
        y = y1 + y2

        X = (X - np.min(X)) / np.max(X) - np.min(X)
        return np.expand_dims(X.astype(np.float32),1), np.expand_dims(y.astype(np.float32),1)

    def generate_tf_dataset(self) -> tf.data.Dataset:
        """convenience function turning X and Y into a tf.data.Dataset"""
        
        dataset = tf.data.Dataset.from_tensor_slices((self.X, self.y))
        dataset = dataset.repeat()
        dataset = dataset.shuffle(self.shuffle)
        dataset = dataset.batch(100)
        return dataset
    
    def generate_tf_dataset_iterator(self):
        """convenience function to generate one shot iterator from tf.data.Dataset.
        tf needs to get rid of these crazy long function calls"""
        return self.generate_tf_dataset().make_one_shot_iterator()
    
    def generate_tf_dataset_iterator_graph(self) -> tf.Graph:
        """returns the tf.Graph of the Dataset"""
        with tf.Graph().as_default() as g_1:
            it = self.generate_tf_dataset_iterator()
            get_next = tf.identity(it.get_next(), name = 'next')
            X_ =  tf.identity(it.get_next()[0], name="X")
            y_ =  tf.identity(it.get_next()[1], name="y")

            return g_1.as_graph_def()