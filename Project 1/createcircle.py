import math as mt
import random
n = 100
for i in range(n):
    ang = random.randint(0,361)
    print (200 + 55 * mt.cos(ang) + random.randint(-100,100)/100.0,300 + 55* mt.sin(ang) + random.randint(-100,100)/100.0)
    #small errors
    #print random.randint(-100,100)/100.0

