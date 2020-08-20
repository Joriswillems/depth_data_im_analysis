from io import BytesIO
from PIL import Image
import gzip, base64, json
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix, vstack
import more_itertools as mi

def get_depth_maps(paths, dt=False, verbose=True):
    
    """
    Retrieve depth maps from tar.gz file. Depth maps
    are hashed with their timestamp and are returned
    as a dictionairy. The dt argument enables returning
    timestamps as datetime objects.
    """
    # create variables for saving the extracted depth maps
    time = []
    depth_maps = []

    # loop over all depth map files in depth_map_dir
    for path in paths:

        # retrieve depth maps (from approx 1 minute of data)
        t, d = __decode_single_depth_map_file(path, dt=dt, verbose=verbose)

        # concatenate to data that has already been collected
        time += t
        depth_maps += d

    # for convenience, we convert to numpy arrays for easier/faster analysis
    time = np.array(time)
    depth_maps = np.stack(depth_maps)
    
    # hash depth maps and return    
    return dict(zip(time, depth_maps))

def __decode_single_depth_map_file(path, dt=False, verbose=True):

    def __decode_single_im(im_encoded):

        sbuf = BytesIO()

        sbuf.write(im_encoded)
        
        im_decoded = np.array(Image.open(sbuf))
        
        sbuf.close()
        
        return im_decoded
    
    def __decode_depth_file(path, dt):
        
        file = gzip.open(path)
        json_file = json.load(file)
        
        if dt: f = lambda x: datetime.strptime(x['time'], "%Y-%m-%d_%H:%M:%S.%f")
        else: f = lambda x: x['time']
        
        data = [(f(v), __decode_single_im(base64.b64decode(v['v']))) for v in json_file]
        
        file.close()
        
        return zip(*data)

    # decode all instantaneous images
    time, depth_maps = __decode_depth_file(path, dt)


    if verbose:
        
        if dt:
            min_time = time[0]
            max_time = time[-1]

        else:
            min_time = datetime.strptime(time[0], "%Y-%m-%d_%H:%M:%S.%f")
            max_time = datetime.strptime(time[-1], "%Y-%m-%d_%H:%M:%S.%f")

        print("{} depth maps retrieved!\nDate: {}\nBetween {} and {}\n".format(len(time), 
                                                                             min_time.date(), 
                                                                             min_time.time(),
                                                                             max_time.time()))

    return  time, depth_maps