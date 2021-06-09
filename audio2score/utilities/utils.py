import os



def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)