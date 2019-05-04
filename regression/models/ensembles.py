import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.getcwd()))

from models.networks import EnsembleNetwork

class EnsembleParent(object):
    """
    https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/05_Ensemble_Learning.ipynb
    """

    def __init__(self, estimator_stats=None, num_epochs=10):
        pass

    def train(self, X, y):
        pass

    def predict(self, X, return_samples=False):
        pass

    def plot(self, X, y, plot_samples=False, original_func=None,
             sorted_index=None):
        preds_dict = self.predict(X, True)
        y_hats = np.array([mean[0] for mean in preds_dict['means']])
        y_stds = np.array([std[0] for std in preds_dict['stds']])
        samples = preds_dict['samples']
        X_plot = [x[0] for x in np.array(X)]  #[sorted_index]]

        #print(y_hats.shape)
        #print(y_stds.shape)
        #print(y.shape)
        #print(np.array(X).shape)
        #X_plot = X_plot[sorted_index]
        y = np.array(y)  #[sorted_index]
        y_hats = y_hats  #[sorted_index]
        y_stds = y_stds  #[sorted_index]

        plt.plot(X, y, 'k*', label='Data', alpha=0.4)

        plt.plot(X_plot, y_hats, label='predictive mean')
        plt.fill_between(X_plot, y_hats + y_stds, y_hats, alpha=.3, color='b')
        plt.fill_between(X_plot, y_hats - y_stds, y_hats, alpha=.3, color='b')

        if plot_samples:
            for i, sample in enumerate(samples):

                plt.plot(X_plot, sample, label='sample {}'.format(i),
                         alpha=max(1 / len(samples), 0.3))
        if original_func:
            y_original = original_func(X_plot)
            plt.plot(X, y_original, label='generating model')
        plt.legend()


class VanillaEnsemble(EnsembleParent):
    """
    https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/05_Ensemble_Learning.ipynb
    """

    def __init__(self,ds_graph = None, estimator_stats=None, num_epochs=10):

        default_ensemble = [{
            'ds_graph' : gdef,
            'num_neurons': [10, 50, 20],
            'num_epochs': num_epochs,
            'seed':42
        }, {
            'ds_graph' : gdef,
            'num_neurons': [100, 10],
            'num_epochs': num_epochs,
            'seed': 43
        }, {
            'ds_graph' : gdef,
            'num_neurons': [5, 150],
            'num_epochs': num_epochs,
            'seed': 44
        }]

        self.estimator_stats = estimator_stats or default_ensemble
        self.estimator_list = [
            EnsembleNetwork(**x) for x in self.estimator_stats
        ]

    def fit(self, epochs = 10):
        '''This is where we build in the Online Bootstrap'''
        for estimator in self.estimator_list:
            #estimator.train_and_evaluate(X, y,shuffle=False)
            estimator.fit(epochs)#(X,y)
        #system('say training  complete')
        
    def train(self,X,y,epochs = 10):
      for epoch in epochs:
        for estimator in self.estimator_list:
          estimator.train(X,y)

    def predict(self, X, return_samples=False):
        pred_list = []
        for estimator in self.estimator_list:
            prediction = estimator.predict(X)
            pred_list.append(prediction)
        #for i,sample in enumerate(pred_list):
        #   assert(np.isnan(sample) is False), 'sample {} contains NaN'.format(i)
        stds = np.std(pred_list, 0)
        means = np.mean(pred_list, 0)
        #assert(np.isnan(stds) is False)
        #assert(np.isnan(means) is False)
        return_dict = {'stds': stds, 'means': means}
        if return_samples:
            return_dict['samples'] = pred_list
        #system('say prediction complete')

        return return_dict


class OnlineBootstrapEnsemble(VanillaEnsemble):
    """
    https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/05_Ensemble_Learning.ipynb
    """
    def __init__(self,gdef, estimator_stats = None,num_estimators=10,num_epochs=10,seed=10):
        
        default_ensemble = [{
            'ds_graph' : gdef,
            'num_neurons': [10, 5, 5, 2],
            'num_epochs': num_epochs,
            'seed':42
        },
            {
            'ds_graph' : gdef,
            'num_neurons': [10, 10, 5],
            'num_epochs': num_epochs,
            'seed': 43
        }, {
            'ds_graph' : gdef,
            'num_neurons': [5, 15, 5],
            'num_epochs': num_epochs,
            'seed': 44
        }
        ]

        self.estimator_stats = estimator_stats or default_ensemble
        self.estimator_list = [
            EnsembleNetwork(**x) for x in self.estimator_stats
        ]
        

    def fit(self, epochs = 10):
      for i in range(epochs*2): # only training in half of the cases
        for estimator in self.estimator_list:
          if np.random.random() > 0.5:
            #estimator.train_and_evaluate(X, y,shuffle=False)
              estimator.fit(epochs)#(X,y)
        #system('say training  complete')

    def train(self,X,y,epochs = 10):
      for epoch in range(epochs*2):
        
        for estimator in self.estimator_list:
          #mask = np.random.choice([0,1], size= len(y)).astype(bool)
          #X = X[mask]
          #y = y[mask]
          estimator.train(X,y)

            
    def predict(self,X,return_samples=False):
        pred_list = []
        for estimator in self.estimator_list:
            prediction = estimator.predict(X)
            pred_list.append(prediction)
            
        stds = np.std(pred_list,0)
        means = np.mean(pred_list,0)
        
        
        return_dict = {'stds':stds,'means':means}
        if return_samples:
            return_dict['samples'] = pred_list
        #system('say prediction complete')

        return return_dict
            
