'''
Created on 18 Mar 2015

@author: MRSTIM003
'''

from point import Point
from Observations import Observations
import numpy as np
import sympy as syp
import math as mt

def dist(Xp,Yp,Yc,Xc):
    return mt.sqrt((Xp-Xc)**2 + (Yp-Yc)**2)

if __name__ == '__main__':
    Points = {}
    Obs = {}
    f = open('Book.csv', 'r')    
    for line in f:
        sp = line.split(',')
        
        nameOfLastP = None
        if sp[0] == 'CP' or sp[0] == 'P':

            Tag = sp[0]
            name = sp[1]
            nameOfLastP = str(sp[1])
            x = float(sp[2])
            y = float(sp[3])
            Points[name] = Point(x, y, Tag)
        if sp[0] == 'OBS':

            Name = str(sp[1]+'-'+sp[2])
            To = sp[2]
            dirN = sp[3]
            Obs[Name] = Observations(To, dirN)
            
            
    '''Point dicts created'''            
#    print (Points)
    #print (Obs)
    
    unknowns = []
    for Point_name, P in Points.items():
        if P.Tag == 'P':
            unknowns.append(Point_name)
    unknowns.sort()

    A = np.zeros(shape=(len(Obs),len(unknowns) ))
    P = np.array(np.identity(len(Obs)))
    
    
    row = 0
    col = 0
    for unknown in unknowns:
            for obs_name,obs in Obs.items():
                sp = obs_name.split('-')
                if sp[0] == unknown:
                    
                    point = Points[unknown]
                    target = Points[sp[1]]
                    
                    dx1,dx2,dy1,dy2 = syp.symbols('dx1 dx2 dy1 dy2', real=True)
                    
                    a = syp.Function('a')
                    x1,x2,y1,y2 = syp.symbols('x1 x2 y1 y2', real=True)
                    a = (x2 - x1)/dist(target.x,target.y,point.x,point.y)
                    
                    b = syp.Function('b')
                    b = (y2 - y1)/dist(target.x,target.y,point.x,point.y)
                    
                    f = syp.Function('f')
                    f = a*(dy1 - dy2) - b*(dx1 - dx2)
                    
                    valueY = f.diff(dy1)
                    valueX = f.diff(dx1)
                    
                    
                    
                    ValueY.subs((y2,target.y)(y1,point.y))
                    print (f.diff(dy1))
                    print (f.diff(dx1))
                    
                    
                    
                row+=1
            col+=1
        
        
        
#    print(A)
        

    '''extraction structure'''
#    for Point_name, P in Points.items():
#        print (Point_name, P.x , P.y )
#        for obs_name,obs in Obs.items():
#            sp = obs_name.split('-')
#            if sp[0] == Point_name:
#                print (sp, obs.Direction)
    print()
    

































