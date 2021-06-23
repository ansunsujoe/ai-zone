import tensorflow as tf
from definitions import CONFIG_DIR
from pathlib import Path
import json

class Config():
    def __init__(self):
        self.dictionary = {}
        
        # Get everything from security config
        data = json.loads(open(Path(CONFIG_DIR) / "security.json", "r").read())
        for key in data:
            self.add(key, data[key])
            
    def get(self, name):
        try:
            return self.dictionary[name]
        except KeyError:
            raise Exception("Config " + str(name) + " is not defined. Please define it by using the add() method.")
    
    def add(self, name, value):
        self.dictionary[name] = value