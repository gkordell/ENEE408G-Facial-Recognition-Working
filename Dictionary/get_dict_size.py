#simply returns the size of the dictionary
import sys
import pickle

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

dict = load_obj("num_names_dict")
print len(dict)

