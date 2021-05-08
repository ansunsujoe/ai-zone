import tensorflow as tf
import logging

class Hyperparams():
    def __init__(self):
        self.dictionary = {}
            
    def get(self, name):
        try:
            return self.dictionary[name]
        except KeyError:
            raise Exception("Hyperparameter " + str(name) + " is not defined. Please define it by using the set() method.")
    
    def add(self, name, value):
        self.dictionary[name] = value