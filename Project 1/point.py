class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def fixPoints(self,dx,dy):
        self.x = x + dx
        self.y = y + dy