import math

from think import Face

FACES = [Face(23.5, 21.5, 13.5, 16.5), Face(19.5, 11.5, 18, 12), Face(23.5, 16.5, 13.5, 12), Face(23.5, 16.5, 18, 12),
             Face(23.5, 16.5, 18, 7.5), Face(15, 11.5, 9, 16.5), Face(19.5, 16.5, 9, 7.5), Face(15, 11.5, 18, 16.5),
             Face(15, 11.5, 18, 7.5), Face(15, 11.5, 9, 7.5), Face(19.5, 21.5, 9, 7.5), Face(23.5, 11.5, 9, 7.5),
             Face(15, 21.5, 13.5, 12), Face(19.5, 16.5, 9, 16.5), Face(19.5, 16.5, 9, 12), Face(15, 16.5, 13.5, 16.5),
             Face(23.5, 11.5, 9, 16.5), Face(15, 11.5, 13.5, 12), Face(15, 21.5, 18, 16.5), Face(19.5, 11.5, 18, 16.5),
             Face(19.5, 21.5, 9, 7.5), Face(15, 21.5, 18, 12), Face(19.5, 11.5, 13.5, 16.5), Face(15, 16.5, 13.5, 12),
             Face(15, 16.5, 18, 16.5), Face(19.5, 21.5, 9, 16.5), Face(15, 16.5, 18, 12), Face(19.5, 16.5, 13.5, 7.5),
             Face(15, 21.5, 13.5, 16.5), Face(15, 11.5, 13.5, 12), Face(23.5, 11.5, 18, 12), Face(19.5, 11.5, 13.5, 12)]

categories = [None for i in range(len(FACES))]
slot_vals = {'eh': [23.5, 19.5, 15], 'es': [21.5, 11.5, 16.5], 'nl': [13.5, 18, 9], 'mh': [24.75, 18.0, 11.25]}
sizes = {0: 'small', 1: 'medium', 2: 'large'}

cat1 = [["small", "large", "medium", "small"],
        ["medium", "small", "large", "medium"],
        ["small", "medium", "medium", "medium"],
        ["small", "medium", "large", "medium"],
        ["small", "medium", "large", "large"]]

cat2 = [["large", "small", "small", "small"],
        ["medium", "medium", "small", "large"],
        ["large", "small", "large", "small"],
        ["large", "small", "large", "large"],
        ["large", "small", "small", "large"]]

for i in range(len(FACES)):
    face = FACES[i]
    cat = []
    for feature, val in face.get_features().items():
        cat.append(sizes[slot_vals[feature].index(val)])
    if cat in cat1:
        categories[i] = 1
    elif cat in cat2:
        categories[i] = 2
    else:
        print(i, cat)

