"""Experimental horizontal reflection of right single hands into the left slot."""
import numpy as np
VERSION='single-hand-reflect-x-left-v1'

def canonicalize(features):
    x=np.asarray(features,dtype=float)
    if x.shape!=(126,) or not np.isfinite(x).all():
        raise ValueError('Expected 126 finite features.')
    out=x.copy()
    left=bool(np.any(x[:63])); right=bool(np.any(x[63:]))
    if right and not left:
        hand=x[63:].reshape(21,3).copy()
        hand[:,0]*=-1
        out[:63]=hand.reshape(-1)
        out[63:]=0
    return out

def canonicalize_batch(features):
    return np.array([canonicalize(x) for x in features])
