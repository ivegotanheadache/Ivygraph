import logging
import networkx as nx
from itertools import combinations

logger = logging.getLogger(__name__)


def get_leaves(G):
    return [n for n in G.nodes if G.out_degree(n) == 0]


def compute_heights(G):
    heights = {}
    for node in reversed(list(nx.topological_sort(G))):
        children = list(G.successors(node))
        heights[node] = 1 if not children else 1 + max(heights[c] for c in children)
    return heights

def find_similar_leaves(G, root, threshold=0.7, min_depth=2):
    leaves = get_leaves(G)
    depths = nx.single_source_shortest_path_length(G, root)
    heights = compute_heights(G)

    # Precompute una volta sola per leaf, non per coppia
    leaf_ancestors = {leaf: nx.ancestors(G, leaf) for leaf in leaves}

    ancestor_to_leaves = {}
    for leaf in leaves:
        for ancestor in leaf_ancestors[leaf]:
            if heights[ancestor] >= min_depth:
                ancestor_to_leaves.setdefault(ancestor, set()).add(leaf)

    seen = set()

    def _scored_pairs():
        for grouped_leaves in ancestor_to_leaves.values():
            if len(grouped_leaves) < 2:
                continue
            for a, b in combinations(sorted(grouped_leaves), 2):
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                ancs_a, ancs_b = leaf_ancestors[a], leaf_ancestors[b]
                small, large = (ancs_a, ancs_b) if len(ancs_a) <= len(ancs_b) else (ancs_b, ancs_a)
                common = (n for n in small if n in large)
                lcs_node = max(common, key=lambda n: depths.get(n, -1), default=None)
                if lcs_node is None:
                    continue
                score = 2 * depths[lcs_node] / (depths[a] + depths[b])
                if score >= threshold:
                    yield -score, a, b

    return [
        {"words": [a.split('.')[0], b.split('.')[0]], "wp_distance": -neg_score}
        for neg_score, a, b in sorted(_scored_pairs())
    ]