#arg1 is the username, returns named tuple ID with number and pass
import sys
from collections import namedtuple
import pickle

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

ID = namedtuple("ID",["number","password"])

dict = load_obj("names_info_dict")

if __name__ == "__main__":
	if(sys.argv[1] in dict):
		print dict[sys.argv[1]]
	else:
		print ("Not in dict")