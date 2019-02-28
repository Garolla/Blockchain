class Photo:

    isVertical = True
    tags = []

    def __init__(self, isVertical, tags):
        self.isVertical = isVertical
        self.tags = tags


class Slide:

    tags = []

    def __init__(self, horizontalPhoto = Photo()):
        self.isVertical = isVertical
        self.tags = horizontalPhoto.tags

    def __init__(self, verticalPhoto1 = Photo(), verticalPhoto2 = Photo()):
            self.tags = verticalPhoto1.tags + verticalPhoto2.tags
            self.tags = list(dict.fromkeys(self.tags))
            print(self.tags)