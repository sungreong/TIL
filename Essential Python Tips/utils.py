import logging
import os

def make_dirs(path , local_path = []) :
    if tf.io.gfile.exists(path) :
        tf.io.gfile.rmtree(path)
        tf.io.gfile.makedirs(path)
    else :
        tf.io.gfile.makedirs(path)
    folder_name = [path]
    for local in local_path :
        local_folder = os.path.join(path, local)
        folder_name.append(local_folder)
        tf.io.gfile.makedirs(local_folder)
    return folder_name


def loggingmaker(name, path) :
    log = logging.getLogger(name)
    log.handlers = []  
    log.propagate = True
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(message)s","%Y-%m-%d %H:%M")
    # os.path.join(os.getcwd(), LogPath)
    fileHandler = logging.FileHandler(path, mode="w")
    streamHandler = logging.StreamHandler()
    fileHandler.setLevel(logging.DEBUG);fileHandler.setFormatter(formatter)
    streamHandler.setLevel(logging.DEBUG);streamHandler.setFormatter(formatter)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)
    return log
