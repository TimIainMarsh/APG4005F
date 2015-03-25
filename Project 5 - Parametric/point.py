class Point(object):
    def __init__(self,x,y,Tag):
        self.x = x
        self.y = y
        self.Tag = Tag
    def updateX(self,dx):
        self.x += dx
    def updateY(self,dy):
        self.y += dy