import os
import os.path
from params import FILTER_PATH



def create_path(name):
    folder = name
    if not os.path.isdir(FILTER_PATH):
        os.mkdir(FILTER_PATH)
        
    os.chdir(FILTER_PATH)
    
    if not os.path.isdir(folder):
        os.mkdir(folder)        
    os.chdir("../")
