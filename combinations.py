from itertools import permutations
from graph import find_synonyms
from config import MIN_CHAR_TOTAL, MAX_CHAR_TOTAL
import re

#logger = logging.getLogger(__name__)

def get_perm(comb):
    A = find_synonyms(comb[0])
    A.append(comb[0])
    B = find_synonyms(comb[1])
    B.append(comb[1])
    print(comb, A, B)
    for a in A:
        a_clean = re.sub(r'[^A-Za-z0-9\s]', '', a)
        for b in B:
            b_clean = re.sub(r'[^A-Za-z0-9\s]', '', b)
            yield [a_clean, b_clean]

def variances(combs, skip_short=True, skip_long=False):
    for combination in combs:
        alphabet = [
            word
            for i in combination
            for word in i.split(' ')
            if (not skip_short or len(word) > 2)
            and (not skip_long or len(word) < 11)
        ]
        if not alphabet:
            continue

        sorted_lengths = sorted(len(w) for w in alphabet)
        seen = set() 

        for j in range(1, len(alphabet) + 1):
            # lunghezza minima raggiungibile concatenando i j token più corti
            min_possible_len = sum(sorted_lengths[:j])
            if min_possible_len >= MAX_CHAR_TOTAL:
                break  

            for p in permutations(alphabet, j):
                word = ''.join(p).capitalize()
                if MIN_CHAR_TOTAL < len(word) < MAX_CHAR_TOTAL and word not in seen:
                    seen.add(word)
                    yield word


