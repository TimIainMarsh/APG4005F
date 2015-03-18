import numpy as np
import math as mt
import scipy
import point as pt
import random

def createTransformPoints():
    n = 10
    coords1 = []
    coords2 = []
    for i in range(n):

        X =random.randint(-10,10)
        Y =random.randint(-10,10)
        point = np.matrix([[X],
                           [Y]])
        theta = mt.radians(45)
        rotMX = np.matrix([[mt.cos(theta), -mt.sin(theta)],
                           [mt.sin(theta), mt.cos(theta)]])
        transformed_point = 1.02 * rotMX * point + np.matrix([[2],
                                                               [2]])
        
        coords1.append(point)
        coords2.append(transformed_point)
    return coords1, coords2



C1,C2 = (createTransformPoints())


'''initial estimates for circle radius and center'''
Tx = 0
Ty = 0
scale = 1.02
To = 0
''''''

for i in range (10):

    A =  np.matrix([[0,0,0,0],])
    A = np.delete(A,0, axis=0)

    B =  np.zeros(shape=(len(C1)*2,len(C1)*4))
    
    P = np.matrix(np.identity(len(C1)*4))

    w =  np.matrix([[0],])
    w = np.delete(w,0, axis=0)
    
    vert = 0
    hor = 0
    
    for row in range(len(C1)):
        X1 = float((C1[row])[0])
        Y1 = float((C1[row])[1])
        
        X2 = float((C2[row])[0])
        Y2 = float((C2[row])[1])

        
        ''' Making A '''
        As1 =  X1*mt.cos(To) - Y1*mt.sin(To)
        At1 =  - scale * X1 * mt.sin(To) - scale * Y1 * mt.cos(To)
        Atx1 = 1
        Aty1 = 0
        
        As2 =  X1*mt.sin(To) + Y1*mt.cos(To)
        At2 =  scale * X1 * mt.cos(To) - scale * Y1 * mt.sin(To)
        Atx2 = 0
        Aty2 = 1
        
        Arow1 = [As1 , At1, Atx1, Aty1]
        Arow2 = [As2 , At2, Atx2, Aty2]
        A = np.vstack([A,Arow1])
        A = np.vstack([A,Arow2])


        ''' Making B '''
        Bx1_1 = scale * mt.cos(To)
        By1_1 = - scale * mt.sin(To)
        Bx2_1 = -1
        By2_1 = 0
        
        Bx1_2 = scale * mt.sin(To)
        By1_2 = scale * mt.cos(To)
        Bx2_2 = 0
        By2_2 = -1
        
        
        B[vert][hor] = Bx1_1
        B[vert][hor+1] = By1_1
        B[vert][hor+2] = Bx2_1
        B[vert][hor+3] = By2_1
        
        B[vert+1][hor] = Bx1_2
        B[vert+1][hor+1] = By1_2
        B[vert+1][hor+2] = Bx2_2
        B[vert+1][hor+3] = By2_2
        
        vert += 2
        hor += 4        


        ''' Making w '''

        c1 = (X1*mt.cos(To) - Y1*mt.sin(To) + Tx) - X2
        c2 = (X1*mt.sin(To) + Y1*mt.cos(To) + Ty) - Y2
        w = np.vstack([w,c1])
        w = np.vstack([w,c2])


    P = np.asmatrix(P)
    B = np.asmatrix(B)

    Q = (B * P.I * B.T).I
    
    X = -(A.T * Q * A).I * A.T * Q * w
    print (X)
    K = - Q * (A*X + w)
    
    V = P.I * B.T * K

    scale = scale + float(X[0])
    To = To + float(X[1])
    Tx = Tx + float(X[2])
    Ty = Ty + float(X[3])
    
    
    
'''Error in Scale'''
print('scale',scale)
print('To',mt.degrees(To))
print('Tx',Tx)
print('Ty',Ty)


