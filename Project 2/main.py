import numpy as np
import math as mt
import scipy
import point as pt
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
    return mt.sqrt((Xp-Xc)**2 + (Yp-Yc)**2)

#initial estimates for circle radius and center
Xo,Yo = 205.0,305.0
Ro = 54.0

for i in range (10):

    A =  np.matrix([[0,0,0],])
    A = np.delete(A,0, axis=0)

    B =  np.zeros(shape=(100,200))
    
    P = np.matrix(np.identity(200))

    w =  np.matrix([[0],])
    w = np.delete(w,0, axis=0)
    vert = 0
    hor = 0
    for row in range(len(Points)):
        Xp = float(Points[row].x)
        Yp = float(Points[row].y)


        Ax =  -2*(Xp-Xo)
        Bx = 2*(Xp-Xo)
        Ay =  -2*(Yp-Yo)
        By = 2*(Yp-Yo)
        Ar =  -2*Ro
        Arow = [Ax , Ay, Ar]
        A = np.vstack([A,Arow])
        
        c =(Xp - Xo)**2 + (Yp - Yo)**2 - Ro**2
        w = np.vstack([w,c])
        
        B[vert][hor] = Bx
        B[vert][hor+1] = By
        vert += 1
        hor += 2
        
    Q = (B * P.I * B.T).I
    
    X = -(A.T * Q * A).I * A.T * Q * w
#    print(X)
    K = - Q * (A*X + w)
    
    V = P.I * B.T * K

    Xo = Xo + float(X[0])
    Yo = Yo + float(X[1])
    Ro = Ro + float(X[2])

print('Xo',Xo)
print('Yo',Yo)
print('Ro',Ro)



