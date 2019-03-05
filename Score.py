from Photo import Photo
from Slide import Slide



# Python program to illustrate the intersection
# of two lists using set() method


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def score(sl1,sl2):
    inters = intersection(sl1.tags,sl2.tags)
    intersCount = len(inters)
    if intersCount == 0:
        return 0

    diffA = set(sl1.tags).difference(set(sl2.tags))
    diffB = set(sl2.tags).difference(set(sl1.tags))
    return min(intersCount,len(diffA),len(diffB))


def totalScore(slides):
    total = 0
    i = 0
    while i < len(slides) - 1:
        total += score(slides[i], slides[i+1])
        i += 1
    return total