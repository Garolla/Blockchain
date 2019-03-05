from Photo import Photo

class Slide:

    ident = -1
    photo = []
    tags = []

    def __init__(self, photos):
        if len(photos) > 1:
            self.photo = photos
            self.tags = photos[0].tags + photos[1].tags
            self.tags = list(dict.fromkeys(self.tags))
            #print(self.photo[0].ind)
        else:
            self.photo = photos
            self.tags = photos[0].tags
            #print(self.photo[0].ind)

    def __eq__(self, other):
        common_elements = set(self.photo) & set(other.photo)
        return len(self.photo) == len(common_elements)


    def __hash__(self):
        phIds = ""
        for p in self.photo:
            phIds = phIds + p.ind + " "

        return hash(phIds)

    def __str__(self):
        phIds = ""
        for p in self.photo:
            phIds = phIds + p.ind + " "
        return phIds