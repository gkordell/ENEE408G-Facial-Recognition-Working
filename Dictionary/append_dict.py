#arg1 is num, arg2 is name, arg3 is pass, adds info to dicts
import sys
from collections import namedtuple
import pickle

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

ID = namedtuple("ID",["number","password"])

names_info_dict = load_obj("names_info_dict")
num_names_dict = load_obj("num_names_dict")

if __name__ == "__main__":
	num_names_dict[int(sys.argv[1])] = sys.argv[2]
	names_info_dict[sys.argv[2]] = ID(sys.argv[1],sys.argv[3])
	save_obj(num_names_dict,"num_names_dict")
	save_obj(names_info_dict,"names_info_dict")
	'''
	if(sys.argv[1] in dict):
		print dict[sys.argv[1]]
	else:
		print ("Not in dict")
	'''