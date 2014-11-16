import matplotlib.pylab as plt
import numpy as np


def _hex_points (center,scale):    
    xc,yc = center
    xs,ys = scale    
    dx = xs*0.5
    dy = ys/3.0
       
    xmin = xc-dx
    xmax = xc+dx
    
    ymin = yc-2.0*dy 
    y1 = ymin + dy 
    # yc
    y2 = yc + dy 
    ymax = y2 + dy 
    
    pts = np.zeros((6,2))
    pts[0] = xc,ymin 
    pts[1] = xmin,y1
    pts[2] = xmin,y2
    pts[3] = xc,ymax 
    pts[4] = xmax,y2 
    pts[5] = xmax,y1 
    return pts 
    
def plot_a2lattice (a2lattice):
    scale = a2lattice.scale 
    center = a2lattice.center 
    
 
    
    
    
    