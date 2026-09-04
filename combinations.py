from itertools import permutations
from graph import find_synonyms
from config import MIN_CHAR_TOTAL, MAX_CHAR_TOTAL

def get_perm(comb):
    A = find_synonyms(comb["words"][0])
    B = find_synonyms(comb["words"][1])
    print(comb, A, B)
    for a in A:
        for b in B:
            yield {"combination": [str(a), str(b)], "wp_distance": comb["wp_distance"]}

def variances(combs, skip_short=True, skip_long=False):
    for combination in combs:
        sett = combination['combination']
        alphabet = [
            word
            for i in sett
            for word in i.split(' ')
            if (not skip_short or len(word) > 2)
            and (not skip_long or len(word) < 11)
        ]
        for j in range(1, len(alphabet) + 1):
            for p in permutations(alphabet, j):
                word = ''.join(p).capitalize()
                if len(word) > MIN_CHAR_TOTAL and len(word) < MAX_CHAR_TOTAL:
                    yield word

def new(p, num_min=1, num_max=100, step=1):
    yield p
    for j in range(num_min, num_max, step):
        yield p + str(j)
