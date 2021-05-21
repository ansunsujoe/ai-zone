import tensorflow as tf
import logging

class Config():
    def __init__(self):
        self.dictionary = {}
            
    def get(self, name):
        try:
            return self.dictionary[name]
        except KeyError:
            raise Exception("Config " + str(name) + " is not defined. Please define it by using the add() method.")
    
    def add(self, name, value):
        self.dictionary[name] = value