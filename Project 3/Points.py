'''
Created on 04 Mar 2015

@author: MRSTIM003
'''
from Point import Point
class Points(dict):


    def __init__(self):
        print()
        
    def read(self, filename):
        f = open(filename, 'r')    
        f.readline()
        number = 1
        for line in f:
            sp = line.split(',')
            
            x = float(sp[0])
            y = float(sp[1])
            z = float(sp[2])
            self[number] = Point(x, y, z)
            number+=1
            
            
            
            