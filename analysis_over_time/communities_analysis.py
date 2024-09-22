import networkx as nx
import numpy as np
import pandas as pd
import itertools

from data.datasets import VoteResultType
from milestone_communities.main import pivot_results, compute_connections, detect_communities


def _get_communities(votes_matrix: np.array) -> list[set[int]]:

    connections_matrix = compute_connections(votes_matrix)

    assert connections_matrix.shape[0] == connections_matrix.shape[1]

    idx = list(range(connections_matrix.shape[0]))

    G: nx.Graph = nx.Graph()
    G.add_nodes_from(idx)
    G.add_weighted_edges_from(
        ((idx[i], idx[j], connections_matrix[i, j]) for i, j in itertools.product(
            range(len(idx)), repeat=2) if i < j))

    return detect_communities(G)


def _generate_connection_matrix(votes: pd.DataFrame) -> tuple[np.array, list[int]]:
    pivot, idx_to_vip_id = pivot_results(
        votes,
        allowed_vote_values={VoteResultType.FOR.value, VoteResultType.AGAINST.value},
        min_votes=15,
    )

    # +/-1 for support/oppose, 0 otherwise
    as_tristate = np.zeros_like(pivot, dtype=int) + (pivot == VoteResultType.FOR.value) - (
            pivot == VoteResultType.AGAINST.value)
    return as_tristate, idx_to_vip_id


def _compute_cosine_dist(votes_matrix: np.array, set1: set[int], set2: set[int]) -> float:
    votes_matrix = votes_matrix.astype(np.float64)
    avg_vote1 = votes_matrix[list(set1), :].sum(axis=0)
    avg_vote2 = votes_matrix[list(set2), :].sum(axis=0)
    un_normalized = avg_vote1 @ avg_vote2 / (np.sqrt(avg_vote1 @ avg_vote1) * np.sqrt(avg_vote2 @ avg_vote2))
    return (un_normalized + 1) / 2


def compute_communities_cosine_score(votes: pd.DataFrame) -> float:
    vote_matrix, idx_to_vip_id = _generate_connection_matrix(votes)
    communities = sorted(_get_communities(vote_matrix), key=lambda community: len(community), reverse=True)

    coalition, opposition = communities[:2]
    return 1 - _compute_cosine_dist(vote_matrix, coalition, opposition)
