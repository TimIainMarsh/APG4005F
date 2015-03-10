'''
Created on 04 Mar 2015

@author: MRSTIM003
'''

import time
Totstart = time.time()

import math as mt
import random
from Points import Points
from Plotting import Plotting
import numpy as np

def DomeGenerate(filename):
    f = open(filename , 'w')
    n = 400

    for i in range(n):
        r = 37.0
        xof = 100.0
        yof = 200.0
        zof = 100.0
        ang1 = mt.radians(random.randint(0,361))
        ang2 = mt.radians(random.randint(0,91))
        X = xof +  r * mt.cos(ang1) * mt.sin(ang2)+ random.randint(-100,100)/1000.0
        Y = yof +  r * mt.sin(ang1) * mt.sin(ang2)+ random.randint(-100,100)/1000.0
        Z = zof +  r * mt.cos(ang2)+ random.randint(-100,100)/100.0
        line = str(X) + "," + str(Y)  + "," + str(Z) + '\n'
        f.write(line)

def chi_square(A,Ex):
    from scipy.stats import chi2
    
    popVar = (3.0*((3.0/1000)**2.0))
    samVar = float(Ex[3,3])
    df = len(A) * 3 -12
    
    chi_test = df * (samVar/popVar)
    
    sigLevel = 0.05 # - - - - - 1%
    
    mean,var,skew,kurt = chi2.stats(df,moments = 'mvsk')
    chi = chi2.ppf(1-sigLevel,df)
    
    if chi_test > chi:
        print ('Reject null h. Sig Lev: ' + str(sigLevel))
    else:
        print ('Do Not Reject null h. Sig Lev: ' + str(sigLevel))
    
    
    
    
    

if __name__ == '__main__':
#    DomeGenerate("Data.csv")
    
    '''Reading the points out of the point file'''
    '''enter the file name'''
    Points = Points()
    Points.read('data_assignment1.csv')
    

    Xo,Yo,Zo = 0.0,0.0,0.0
    Ro = 5.0
#    Plotting = Plotting()
#    Plotting.plotPoints(Points)
    end = start = 0
    for i in range (10):
#        print ("start it:  " + str(end - start))
        
        start = time.time() ## start of loop
        A =  np.matrix([[0,0,0,0],])
        A = np.delete(A,0, axis=0)
        
        B =  np.zeros(shape=(len(Points),len(Points)*3))
        
        P = P = np.matrix(np.identity(len(Points)*3))
        
        w =  np.matrix([[0],])
        w = np.delete(w,0, axis=0)
        
        vert = 0
        hor = 0
    
        for row in range(len(Points)):
            
            Xp = Points[row+1].x
            Yp = Points[row+1].y
            Zp = Points[row+1].z
            
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

#        K = - Q * (A*X + w)
        
#        V = P.I * B.T * K
    
        Xo = Xo + float(X[0])
        Yo = Yo + float(X[1])
        Zo = Zo + float(X[2])
        Ro = Ro + float(X[3])
        print ('Xo: ',Xo)
        print ('Yo: ',Yo)
        print ('Zo: ',Zo)
        print ('Ro: ',Ro)
#        end = time.time() ##end of loop
        
#    print (X)
    print ('Xo: ',Xo)
    print ('Yo: ',Yo)
    print ('Zo: ',Zo)
    print ('Ro: ',Ro)

    V = A*X - w

    apriori = (V.T * V)/(len(w) - len(X))
    
    Qx = (A.T * A).I
    Ex = float(apriori) * Qx

    Ql = A * Qx * A.T
    EL = float(apriori) * Ql

    P = np.matrix(np.identity(len(w)))
    Ev = float(apriori) * (P.I - Ql)

    '''Running Hypothesis test'''
    chi_square(A,Ex)

    




    Totend = time.time()
#    print ("Total Time: " + str(Totend - Totstart))
    
    
'''
results: 

Xo:  0.000247034040027427
Yo:  0.002073032848955772
Zo:  0.031484892325592366
Ro:  3.084421992716166
Do Not Reject null h. Sig Lev:0.05


Xo:  0.0002470340400274099
Yo:  0.002073032848955792
Zo:  0.03148489232559259
Ro:  3.084421992716166
Do Not Reject null h. Sig Lev: 0.05


Xo:  0.00024703404002740504
Yo:  0.0020730328489557935
Zo:  0.0314848923255926
Ro:  3.084421992716166
Do Not Reject null h. Sig Lev: 0.01


------------------------------
Xo:  0.0002543399319671208
Yo:  0.0020350776362895286
Zo:  0.03143964593823502
Ro:  3.4512850323573794

Xo:  0.000246866976236784
Yo:  0.0020721541758914083
Zo:  0.031484802904184594
Ro:  3.1039203696170934

Xo:  0.00024703870021425647
Yo:  0.0020730121133359205
Zo:  0.03148489159824359
Ro:  3.084483236439781

Xo:  0.00024703393173176153
Yo:  0.0020730323607532804
Zo:  0.031484892310224326
Ro:  3.084421993340411

Xo:  0.0002470340430091016
Yo:  0.002073032837420004
Zo:  0.0314848923252366
Ro:  3.0844219927165475

Xo:  0.0002470340399573045
Yo:  0.002073032848684187
Zo:  0.031484892325583894
Ro:  3.084421992716175

Xo:  0.00024703404002932277
Yo:  0.0020730328489495316
Zo:  0.03148489232559219
Ro:  3.084421992716166

Xo:  0.0002470340400274299
Yo:  0.0020730328489556443
Zo:  0.03148489232559223
Ro:  3.084421992716166

Xo:  0.0002470340400274155
Yo:  0.0020730328489557445
Zo:  0.03148489232559228
Ro:  3.084421992716166

Xo:  0.00024703404002740623
Yo:  0.002073032848955805
Zo:  0.03148489232559233
Ro:  3.084421992716166

Xo:  0.00024703404002740623
Yo:  0.002073032848955805
Zo:  0.03148489232559233
Ro:  3.084421992716166

Do Not Reject null h. Sig Lev: 0.05

'''
