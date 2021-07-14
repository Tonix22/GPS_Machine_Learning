import os
import os.path


def create_path(name):
    folder = name
    os.chdir("./filtered")
    print("current dir is: %s" % (os.getcwd()))

    if not os.path.isdir(folder):
        os.mkdir(folder)        
    os.chdir("../")
