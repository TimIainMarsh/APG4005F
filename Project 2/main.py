
import numpy as np
import math as mt
filename = 'random_points.csv'

f = open(filename , 'r')

number = 0
from point import Point
Points = {}
for line in f:
    sp = line.split(',')
    Points[number] = Point(sp[0],sp[1])
    number+=1
    
def dist(Xp,Xc,Yp,Yc):
    #print mt.sqrt((Xp-Xc)**2 + (Yp-Yc)**2)
    return mt.sqrt((Xp-Xc)**2 + (Yp-Yc)**2)

#initial estimates for circle radius and center
Xo,Yo = 206.0,305.0
Ro = 56.0
for i in range (10):

    A =  np.matrix([[0,0,0],])
    A = np.delete(A,0, axis=0)
    l =  np.matrix([[0],])
    l = np.delete(l,0, axis=0)
    
    for row in range(len(Points)):
        
        Xp = float(Points[row].x)
        Yp = float(Points[row].y)
        sqrt =(Ro **2) - (Xp - Xo)**2
        
        if (sqrt)> 0 and Yp>Yo:
            
            x  = (Xp - Xo)/mt.sqrt(sqrt)
            
            y = 1
            
            r = Ro/mt.sqrt(sqrt)
            
            A = np.vstack([A,[x , y, r]])
            c =mt.sqrt(sqrt) +Yo
            l = np.vstack([l,Yp - c])
            
            
    X = (A.T * A).I * (A.T * l)
    Xo = float(Xo + X[0])
    Yo = float(Yo + X[1])
    estRad = float(Ro + X[2])
#print (l)
print (Xo)
print (Yo)
print (estRad)
#print(len(l))
V = A*X - l
#print(V)
apriori = (V.T * V)/(len(l) - len(X))

print (apriori)
Qx = (A.T * A).I
Ex = float(apriori) * Qx
#print (Ex)
Ql = A * Qx * A.T
EL = float(apriori) * Ql
#print (EL)
P = np.matrix(np.identity(len(l)))
Ev = float(apriori) * (P.I - Ql)
#print (Ev)
