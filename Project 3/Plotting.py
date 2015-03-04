'''
Created on 04 Mar 2015

@author: MRSTIM003
'''
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
        
class Plotting:
    '''
    classdocs
    '''


    def __init__(self):
        print ('Here')
        self.X = []
        self.Y = []
        self.Z = []
    

        
    def plotPoints(self,list):

        for row in range(len(list)):
            Xp = list[row+1].x
            Yp = list[row+1].y
            Zp = list[row+1].z
            self.X.append(Xp)
            self.Y.append(Yp)
            self.Z.append(Zp)
        

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        ax.plot_trisurf(self.X, self.Y, self.Z, cmap=cm.jet, linewidth=0.01)
        
        plt.show()