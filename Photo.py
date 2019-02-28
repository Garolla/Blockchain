class Photo:

    isVertical = True
    tags = []

    def __init__(self, isVertical, tags):
        self.isVertical = isVertical
        self.tags = tags


class Slide:

    tags = []
    photo = []

    def __init__(self, horizontalPhoto = Photo()):
        self.isVertical = isVertical
        self.tags = horizontalPhoto.tags



    def __init__(self, verticalPhoto1 = Photo(), verticalPhoto2 = Photo()):
            self.photo.append(verticalPhoto1)
            self.photo.append(verticalPhoto2)
            self.tags = verticalPhoto1.tags + verticalPhoto2.tags
            self.tags.