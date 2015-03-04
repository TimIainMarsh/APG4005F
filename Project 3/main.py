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
    for i in range (4):
        print ("start it")
        A =  np.matrix([[0,0,0,0],])
        A = np.delete(A,0, axis=0)
        
        B =  np.zeros(shape=(len(Points),len(Points)*3))
        
        P = P = np.matrix(np.identity(len(Points)*3))
        
        w =  np.matrix([[0],])
        w = np.delete(w,0, axis=0)
        
        vert = 0
        hor = 0
    
        for row in range(len(Points)):
    #        print (Points[row+1])
            
            Xp = float(Points[row+1].x)
            Yp = float(Points[row+1].y)
            Zp = float(Points[row+1].z)
            
            Ax = -2*(Xp-Xo)
            Ay = -2*(Yp-Yo)
            Az = -2*(Zp-Zo)
            Ar = -2*Ro
            
            Bx =  2*(Xp-Xo)
            By =  2*(Yp-Yo)
            Bz =  2*(Zp-Zo)
            
            Arow = [Ax , Ay, Az, Ar]
            A = np.vstack([A,Arow])
        
        
            c =(Xp - Xo)**2 + (Yp - Yo)**2 + (Zp - Zo)**2 - Ro**2
            w = np.vstack([w,c])
        
            B[vert][hor] = Bx
            B[vert][hor+1] = By
            B[vert][hor+2] = Bz
            vert += 1
            hor += 3
        
        
        Q = (B * P.I * B.T).I
        
        X = -(A.T * Q * A).I * A.T * Q * w

        K = - Q * (A*X + w)
        
        V = P.I * B.T * K
    
        Xo = Xo + float(X[0])
        Yo = Yo + float(X[1])
        Zo = Zo + float(X[2])
        Ro = Ro + float(X[3])
    
    print (Xo,Yo,Zo,Ro)  
    
    
    