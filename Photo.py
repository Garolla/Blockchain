class Photo:

    ind = -1
    isVertical = True
    tags = []

    def __init__(self, isVertical, tags, ind):
        self.ind = ind
        self.isVertical = isVertical
        self.tags = tags
        #print("PHOTOINDEX")
        #print(str(self.ind))

    def __eq__(self, other):
        return self.ind == other.ind

    def __hash__(self):
        return hash(self.ind)

    def __str__(self):
        return self.ind
