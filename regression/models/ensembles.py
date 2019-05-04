import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.getcwd()))

from models.networks import EnsembleNetwork, CopyNetwork

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
            

class BootstrapThroughTimeBobStrap(VanillaEnsemble):
    #TODO: decide if replace every epoch or meta-epoch
    #TODO: Early stopping if error does not decrease

    def __init__(self,gdef, num_features=None, num_epochs=10, num_estimators=3,
                 model_name='copynetwork', seed=42, num_neurons=[10, 5, 3],
                 initialisation_scheme=None, activations=None,
                 estimator_stats = None):
        
        self.model_name = 'checkpoints/' + model_name
        self.activations = activations
        self.model = CopyNetwork(ds_graph=gdef,
            seed=seed, initialisation_scheme=initialisation_scheme,
            activations=activations, num_neurons=num_neurons,
            num_epochs=num_epochs)
        self.train_iteration = 0

#         super(BootstrapThroughTimeBobStrap, self).__init__(self,gdef, estimator_stats = None,num_estimators=10,num_epochs=10,seed=10)
        self.num_epochs = num_epochs
        self.num_models = num_estimators
        self.initialise_ensemble()

    def initialise_ensemble(self):
        """create list of checkpoint ensembles"""
        name = self.model_name + 'checkpoint_' + str(self.train_iteration)
        self.model.save(name)
        self.checkpoints = [name]

    def get_prediction_list(self, X):
        prediction_list = []
        for ckpt in self.checkpoints:
            self.model.load(ckpt)
            prediction_list.append(self.model.predict(X))

        return prediction_list
    
    def get_prediction_list(self, X):
        prediction_list = []
        for ckpt in self.checkpoints:
            self.model.load(ckpt)
            prediction_list.append(self.model.predict(X))

        return prediction_list
    
#     def predict(self, X):
#         pred_list = self.get_prediction_list(X)
#         self.model.load(self.checkpoints[-1])
#         return self.model.predict(X)
#         #return predictive_uncertainty
        
    def predict(self,X,return_samples=False):
        pred_list = self.get_prediction_list(X)

        stds = np.std(pred_list,0)
        means = np.mean(pred_list,0)
        
        
        return  {'stds':stds,'means':means}

    def fit(self, epochs, burn_in=3):
        """trains the most recent model in checkpoint list and replaces the oldest checkpoint if enough checkpoints exist"""
        ####NEWWWWW
        print('doing a burn in of {} epochs'.format(str(burn_in)))
        if self.train_iteration == 0:
            for i in range(burn_in):
                self.model.load(self.checkpoints[-1])  #load most recent model
                self.model.fit(epochs)
                name = self.model_name + '_burn_in_model_{}'.format(
                    str(burn_in))
                self.model.save(name)
                self.checkpoints = [name]
                
        self.train_iteration += 1
        name = self.model_name + '_checkpoint_' + str(self.train_iteration)

        self.model.load(self.checkpoints[-1])  #load most recent model
#         #####OLDDDD
        for i in range(epochs):

            self.model.fit(epochs)  #train most recent model
        self.model.save(name)  #save newest model as checkpoint
        self.checkpoints.append(name)  #add newest checkpoint

        if len(self.checkpoints
            ) > self.num_models:  #if we reached max number of stored models
                self.checkpoints.pop(0)  #delete oldest checkpoint


class ForcedDiversityBootstrapThroughTime(BootstrapThroughTimeBobStrap):
    # TODO: try out 'burn in' Phase.
    def __init__(self, num_features=None, num_epochs=1, num_models=10,
                 model_name='diversitycopynetwork', seed=42,
                 num_neurons=[10, 5, 3], initialisation_scheme=None,
                 activations=None):

        super(ForcedDiversityBootstrapThroughTime, self).__init__(
            num_features=None, num_epochs=num_epochs, num_models=num_models,
            model_name='forceddiversitycopynetwork', seed=seed,
            num_neurons=num_neurons,
            initialisation_scheme=initialisation_scheme,
            activations=activations)

    def fit(self, X, y, X_test=None, y_test=None):
        """trains the most recent model in checkpoint list and replaces the oldest checkpoint if enough checkpoints exist"""
        for i in range(self.num_epochs):
            self.train_iteration += 1
            name = self.model_name + '_checkpoint_' + str(self.train_iteration)

            self.model.load(self.checkpoints[-1])  #load most recent model
            rsme_before = self.model.compute_rsme(X, y)
            self.model.fit(X, y)  #train most recent model
            rsme_after = self.model.compute_rsme(X, y)
            if rsme_before > rsme_after:
                self.model.save(name)  #save newest model as checkpoint
                self.checkpoints.append(name)  #add newest checkpoint

                if len(
                        self.checkpoints
                ) > self.num_models:  #if we reached max number of stored models
                    self.checkpoints.pop(0)  #delete oldest checkpoint


class ForcedDiversityBootstrapThroughTime2(BootstrapThroughTimeBobStrap):
    def __init__(self, num_features=None, num_epochs=1, num_models=10,
                 model_name='diversitycopynetwork', seed=42,
                 num_neurons=[10, 5, 3], initialisation_scheme=None,
                 activations=None):

        super(ForcedDiversityBootstrapThroughTime2, self).__init__(
            num_features=None, num_epochs=num_epochs, num_models=num_models,
            model_name='forceddiversitycopynetwork', seed=seed,
            num_neurons=num_neurons,
            initialisation_scheme=initialisation_scheme,
            activations=activations)

    def fit(self, X, y, X_test=None, y_test=None):
        """trains the most recent model in checkpoint list and replaces the oldest checkpoint if enough checkpoints exist"""
        for i in range(self.num_epochs):
            self.train_iteration += 1
            name = self.model_name + '_checkpoint_' + str(self.train_iteration)

            self.model.load(self.checkpoints[-1])  #load most recent model
            error_vec_before = self.compute_error_vec(X, y)**2
            self.model.fit(X, y)  #train most recent model
            error_vec_after = self.model.compute_error_vec(X, y)**2

            #maybe use self.compute_error!!!!

            kl_divergence = scipy.stats.entropy(error_vec_before,
                                                error_vec_after)
            print(kl_divergence)

            if np.round(kl_divergence, decimals=1) <= 0:
                continue

            self.model.save(name)  #save newest model as checkpoint
            self.checkpoints.append(name)  #add newest checkpoint

            if len(
                    self.checkpoints
            ) > self.num_models:  #if we reached max number of stored models
                self.checkpoints.pop(0)  #delete oldest checkpoint


class ForcedDiversityBootstrapThroughTime3(
        ForcedDiversityBootstrapThroughTime):
    """implements only getting the std from the ensemble, pred_mean is just last prediction"""

    def __init__(self, num_features=None, num_epochs=1, num_models=10,
                 model_name='diversitycopynetwork', seed=42,
                 num_neurons=[10, 5, 3], initialisation_scheme=None,
                 activations=None):

        super(ForcedDiversityBootstrapThroughTime3, self).__init__(
            num_features=None, num_epochs=num_epochs, num_models=num_models,
            model_name='forceddiversitycopynetwork', seed=seed,
            num_neurons=num_neurons,
            initialisation_scheme=initialisation_scheme,
            activations=activations)

    def predict(self, X):
        self.model.load(self.checkpoints[-1])
        return self.model.predict(X)
        #return predictive_uncertainty

    def get_mean_and_std(self, X):
        pred_list = self.get_prediction_list(X)
        self.model.load(self.checkpoints[-1])
        pred_mean = self.model.predict(X)
        #prediction_list.append(self.model.predict(X))
        pred_std = np.std(pred_list, axis=0)
        return pred_mean, pred_std