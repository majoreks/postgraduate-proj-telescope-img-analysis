import numpy as np
import warnings

def getStridedVector(length, Num_pixels):

    Nheight = np.ceil(length / Num_pixels)
    excessheight = int(( - length % Num_pixels ) % Num_pixels)
    heightstride = int(excessheight // (Nheight-1))
    pos = []
    
    for ii in range(int(Nheight)):
        if ii == 0:
            elem = [0, Num_pixels]
        elif ii == Nheight-1:
            elem = [length-Num_pixels, length]
        else:
            origin = pos[-1][1] - heightstride +1
            elem = [origin, origin+Num_pixels]
        pos.append(elem)
        # print(f"element {ii} : {pos[ii][0]} - {pos[ii][1]} \t Npixels = {pos[ii][1]-pos[ii][0]+1} \t pos has {len(pos)} elements")
    return pos