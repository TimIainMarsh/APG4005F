from math import radians

class Observations(object):
    def __init__(self,To, Direction):
        self.To = To
        self.Direction = (float(Direction[0]) + float(Direction[1])/60.0 + float(Direction[2])/3600.0)
