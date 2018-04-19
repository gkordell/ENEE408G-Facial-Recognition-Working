#arg1 is the number, returns the username
import sys
import pickle

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

dict = load_obj("num_names_dict")

if __name__ == "__main__":
	if(int(sys.argv[1]) in dict):
		print dict[int(sys.argv[1])]
	else:
		print ("Not in dict")
