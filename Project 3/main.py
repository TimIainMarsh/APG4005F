'''
Created on 04 Mar 2015

@author: MRSTIM003
'''

import math as mt
import random
from Points import Points
import numpy as np

def DomeGenerate(filename):
    f = open(filename , 'w')
    n = 500
    for i in range(n):
        r = 50
        xof = 200
        yof = 300
        zof = 100
        ang1 = mt.radians(random.randint(0,361))
        ang2 = mt.radians(random.randint(0,91))
        X = xof +  r * mt.cos(ang1) * mt.sin(ang2)+ random.randint(-100,100)/100.0
        Y = yof +  r * mt.sin(ang1) * mt.sin(ang2)+ random.randint(-100,100)/100.0
        Z = zof +  r * mt.cos(ang2)+ random.randint(-100,100)/100.0
        line = str(X) + "," + str(Y)  + "," + str(Z) + '\n'
        f.write(line)
        
def dist(Xp,Xc,Yp,Yc):
    return mt.sqrt((Xp-Xc)**2 + (Yp-Yc)**2)        

if __name__ == '__main__':
    DomeGenerate("Data.csv")
    
    '''Reading the points out of the point file'''
    
    Points = Points()
    
    '''enter the file name'''
    
    Points.read('Data.csv')


    Xo,Yo,Zo = 202.0,302.0,102.0
    Ro = 52.0
    
    A =  np.matrix([[0,0,0,0],])
    A = np.delete(A,0, axis=0)
    
    B =  np.zeros(shape=(len(Points),len(Points)*3))
    
    
    
    w =  np.matrix([[0],])
    w = np.delete(w,0, axis=0)
    
    
    for row in range(len(Points)):
        print(1)
    
    
    
    
    
    
    
    
    
    
    