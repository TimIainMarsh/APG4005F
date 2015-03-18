from math import radians

class Observations(object):
    def __init__(self,To, Direction):
        sp = Direction.split('.')
        self.To = To
        self.Direction = radians(float(sp[0]) + float(sp[1])/60.0 + float(sp[2])/3600.0)
