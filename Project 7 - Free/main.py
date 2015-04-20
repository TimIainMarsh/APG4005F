'''
Created on 18 Mar 2015

@author: MRSTIM003
'''

from point import Point
from Observations import Observations
import numpy as np
import sympy as syp
import math as mt
import scipy.linalg as LA
from timing import log_timing, log_timing_decorator
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.error('Start')

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=300)

def dist(Xp,Yp,Yc,Xc):
    return mt.sqrt((Xp-Xc)**2 + (Yp-Yc)**2)

def join(T,F,obs):
    diff = round(mt.degrees(obs.Direction) - mt.degrees(np.arctan((F.x-T.x)/(F.y-T.y))))
    ret = np.arctan((F.x-T.x)/(F.y-T.y)) + mt.radians(diff)
    return ret

def Read_Points(filename):
    Points = {}
    Obs = {}
    f = open(filename, 'r')    
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

    return Points, Obs


def A_Matrix(Points, Obs,unknowns):
    print("This A matrix is soooo kak!!")
    A = np.zeros(shape=(len(unknowns),len(Obs)))
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
                a = 206264.8 * ((x2 - x1)/dist(target.x,target.y,point.x,point.y)**2)
                        
                b = syp.Function('b')
                b = 206264.8 * ((y2 - y1)/dist(target.x,target.y,point.x,point.y)**2)
        
                f = syp.Function('f')
                f = a*(dy1 - dy2) - b*(dx1 - dx2)
                if UNsp[1] == 'x':
                    valueX = f.diff(dx1)
                    valueX = valueX.subs(y2,target.y).subs(y1,point.y)
                    A[acc][down] =valueX
                elif UNsp[1] == 'y':
                    valueY = f.diff(dy1)
                    valueY = valueY.subs(x2,target.x).subs(x1,point.x)
                    A[acc][down] =valueY
                            
                            
            elif OBsp[1] == UNsp[0]:
                point = Points[OBsp[0]]
                target = Points[OBsp[1]]
                dx1,dx2,dy1,dy2 = syp.symbols('dx1 dx2 dy1 dy2', real=True)
                        
                x1,x2,y1,y2 = syp.symbols('x1 x2 y1 y2', real=True)
                        
                a = syp.Function('a')
                a = 206264.8 * ((x2 - x1)/dist(target.x,target.y,point.x,point.y)**2)
                        
                b = syp.Function('b')
                b = 206264.8 * ((y2 - y1)/dist(target.x,target.y,point.x,point.y)**2)
        
                f = syp.Function('f')
                f = a*(dy1 - dy2) - b*(dx1 - dx2)
                if UNsp[1] == 'x':
                    valueX = f.diff(dx2)
                    valueX = valueX.subs(y2,target.y).subs(y1,point.y)
                    A[acc][down] =valueX
                            
                elif UNsp[1] == 'y':
                    valueY = f.diff(dy2)
                    valueY = valueY.subs(x2,target.x).subs(x1,point.x)
                    A[acc][down] =valueY
            down += 1
        acc += 1
    return A

def l_Matrix(Obs,Points):
    l = []
    for obs_name,obs in Obs.items():
            OBsp = obs_name.split('-')
            fr = Points[OBsp[0]]
            to = Points[OBsp[1]]
            l.append(float(obs.Direction) - float(join(to,fr,obs)))
    return l


def Creating_Unknowns(Points):
    print("The credibility of this program is unknown.")
    unknowns = []
    for Point_name, P in Points.items():
        if P.Tag == 'P':
            unknowns.append(Point_name + '_x')
            unknowns.append(Point_name + '_y')
    unknowns.sort()
    return unknowns

def get_averageXY(Points):
    print("This program is average...")
    xAve = 0
    yAve = 0
    for j,i in Points.items():
        xAve += float(i.x)
        yAve += float(i.y)
    return xAve,yAve

def find_C(Points):
    print("This is probably not the C you were looking for...")
    c = 0
    xAve,yAve = get_averageXY(Points)
    for j,i in Points.items():
        ei = i.x - xAve
        ni = i.y - yAve
        c += (ei**2 + ni**2)
    c = mt.sqrt(c)
    
    print(c)
    return c

def G_Matrix(Points):
    print("This G matrix is most probably a load of shit!")
    xAve,yAve = get_averageXY(Points)
    c = find_C(Points)
    m = len(Points)
    G = []
    for j,i in Points.items():
        ei = i.x - xAve
        ni = i.y - yAve
        
        gRowX = [1/mt.sqrt(m) , 0, -(ni/c), (ei/c)]
        gRowY = [0, 1/mt.sqrt(m),(ei/c), (ni/c)]
        
        G.append(gRowX)
        G.append(gRowY)
    G = np.asmatrix(G)
    
    return G
                
def S_Transform(G,X):
    sI = [[1,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0]]
    
    gtsigi = (G.T*sI*G).I
    X1 = X - (G * gtsigi * G.T * sI * X)
    
    return X1
print("Tim suuuuucks!!!")
if __name__ == '__main__':
    filename = 'Book.csv'
    Points,Obs = Read_Points(filename)
                      
    iterations = 5
    number = 0
    maxX = 0
    with log_timing(str(iterations) + ' iterations:',logger):
        for i in range(iterations):      #number of iterations
            number +=1
            print ('iteration: '+ str(number))
            unknowns = Creating_Unknowns(Points)
            '''creating the A and populating it'''
            A = A_Matrix(Points, Obs,unknowns)
            P = np.array(np.identity(len(Obs)))
            '''W population'''
            l = l_Matrix(Obs,Points)
            '''casting to matrix'''
            l = (np.asmatrix(l)).T
            A = (np.asmatrix(A)).T
            P = np.asmatrix(P)
            ''''''
            N = A.T *P * A
            G = G_Matrix(Points)
            
            eigs = LA.eigh(N)
            G = []
            count  = 0
            for i in eigs[0]:
                #print (float(i))
                if round(float(i),10) == 0:
                   G.append(eigs[1][count])
                count +=1
            G = np.asmatrix(G)
            print(G)
            GGt = G * G.T
            N_ = N + GGt
            Q_ = N_.I
            Qxx = Q_ - GGt
            X = Q_ * A.T * P * l
            print(X)
            '''update coords'''
            if max(X)> maxX:
                maxX = max(X)
            count = 0
            for i in unknowns:
                sp = i.split('_')
                point = Points[sp[0]]
                if sp[1] == 'x':
                    point.updateX(float(X[count]))
                if sp[1] == 'y':
                    point.updateY(float(X[count]))
                count +=1
        '''End of iterations'''
    print("This program was created by a n0000000000b!!!")
   
    for pn,p in Points.items():
        print (pn,round(p.x,3),round(p.y,3), p.Tag)
    print(str(maxX)+'---')
    

    
    #print(Qxx)


#30 150
#60 140









