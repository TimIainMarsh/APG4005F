'''
Created on 18 Mar 2015

@author: MRSTIM003
'''

from point import Point
from Observations import Observations
import numpy as np
import sympy as syp
import math as mt

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=300)

def dist(Xp,Yp,Yc,Xc):
    return mt.sqrt((Xp-Xc)**2 + (Yp-Yc)**2)

def join(T,F,Targ):
    diff = round(mt.degrees(obs.Direction) - mt.degrees(np.arctan((F.x-T.x)/(F.y-T.y))))
    ret = np.arctan((F.x-T.x)/(F.y-T.y)) + mt.radians(diff)
    return ret

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
            dirN = (sp[3],sp[4],sp[5])
            Obs[Name] = Observations(To, dirN)
            
            
    '''Point dicts created'''            
    
        
    for i in range(1):      #number of iterations
        unknowns = []
        for Point_name, P in Points.items():
            if P.Tag == 'P':
                unknowns.append(Point_name + '_x')
                unknowns.append(Point_name + '_y')
        unknowns.sort()
    
        A = np.zeros(shape=(len(unknowns),len(Obs)))
        P = np.array(np.identity(len(Obs)*2))
        '''creating the A and populating it'''
        acc = 0
        for unknown in unknowns:
            down = 0
            UNsp = unknown.split('_')     #UNsp[0] = name ,UNsp[0] = X or Y
            for obs_name,obs in Obs.items():
                OBsp = obs_name.split('-')
                if OBsp[0] == UNsp[0]:
                    point = Points[OBsp[0]]
                    target = Points[OBsp[1]]
                    dx1,dx2,dy1,dy2 = syp.symbols('dx1 dx2 dy1 dy2', real=True)
                    x1,x2,y1,y2 = syp.symbols('x1 x2 y1 y2', real=True)
                    
                    a = syp.Function('a')
                    a = (x2 - x1)/dist(target.x,target.y,point.x,point.y)
                    
                    b = syp.Function('b')
                    b = (y2 - y1)/dist(target.x,target.y,point.x,point.y)
    
                    f = syp.Function('f')
                    f = a*(dy1 - dy2) - b*(dx1 - dx2)
                    if UNsp[1] == 'x':
                        valueX = f.diff(dx1)
                        valueX = valueX.subs(y2,target.y).subs(y1,point.y)
                        A[acc][down] = valueX
                    if UNsp[1] == 'y':
                        valueY = f.diff(dy1)
                        valueY = valueY.subs(x2,target.x).subs(x1,point.x)
                        A[acc][down] = valueY
                        
                        
                if OBsp[1] == UNsp[0]:
                    point = Points[OBsp[0]]
                    target = Points[OBsp[1]]
                    dx1,dx2,dy1,dy2 = syp.symbols('dx1 dx2 dy1 dy2', real=True)
                    
                    x1,x2,y1,y2 = syp.symbols('x1 x2 y1 y2', real=True)
                    
                    a = syp.Function('a')
                    a = (x2 - x1)/dist(target.x,target.y,point.x,point.y)
                    
                    b = syp.Function('b')
                    b = (y2 - y1)/dist(target.x,target.y,point.x,point.y)
    
                    f = syp.Function('f')
                    f = a*(dy1 - dy2) - b*(dx1 - dx2)
                    if UNsp[1] == 'x':
                        valueX = f.diff(dx2)
                        valueX = valueX.subs(y2,target.y).subs(y1,point.y)
                        A[acc][down] = valueX
                        
                    if UNsp[1] == 'y':
                        valueY = f.diff(dy2)
                        valueY = valueY.subs(x2,target.x).subs(x1,point.x)
                        A[acc][down] = valueY
                down += 1
            acc += 1
        '''done A population'''
        '''L population                    change L to W'''
        W = []
        for obs_name,obs in Obs.items():
            OBsp = obs_name.split('-')
            fr = Points[OBsp[0]]
            to = Points[OBsp[1]]
            W.append(float(join(to,fr,obs)) - float(obs.Direction))
        '''W Done'''
            
            
        ''' create B '''
        vert = 0
        hor = 0
        B =  np.zeros(shape=(len(Obs),len(Obs)*2))
        for TF, obs in Obs.items():
            sp = TF.split('-')
            B[vert][hor] = 1
            B[vert][hor+1] = 1
            vert += 1
            hor += 2
        
        print (B)
        
        
        ''''''
        W = (np.asmatrix(W)).T
        A = (np.asmatrix(A)).T
        B = np.asmatrix(B)
        P = np.asmatrix(P)
        
        X = - (A.T * (B* P.I * B.T).I * A).I * (A.T * (B* P.I * B.T).I * W)   # change


        K = -(B* P.I * B.T).I * (A*X + W) 
        V = P.I * B.T * K

        print (V)
        count = 0
        for i in unknowns:
            sp = i.split('_')
            point = Points[sp[0]]
            if sp[1] == 'x':
                point.updateX(float(X[count]))
            if sp[1] == 'y':
                point.updateY(float(X[count]))
            count +=1



    for pn,p in Points.items():
        print (pn,p.x,p.y, p.Tag)














