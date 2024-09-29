from typing import Literal

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
        min_votes=10,
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
    return 1 - ((un_normalized + 1) / 2)


def compute_score(votes: pd.DataFrame, coallition_opposition: pd.DataFrame, func) -> float:
    vote_matrix, idx_to_vip_id = _generate_connection_matrix(votes)

    coalition_vip_ids = list(coallition_opposition[coallition_opposition["is_coalition"] == 1]["kmmbr_id"])
    opposition_vip_ids = list(coallition_opposition[coallition_opposition["is_coalition"] == 0]["kmmbr_id"])
    vip_id_to_idx = {vip_id: idx for (idx, vip_id) in enumerate(idx_to_vip_id)}
    coalition_indices = {vip_id_to_idx[vip_id] for vip_id in coalition_vip_ids if vip_id in vip_id_to_idx}
    opposition_indices = {vip_id_to_idx[vip_id] for vip_id in opposition_vip_ids if vip_id in vip_id_to_idx}

    return func(vote_matrix, coalition_indices, opposition_indices)


def compute_phi_score(votes_matrix: np.array, set1: set[int], set2: set[int]) -> float:
    def single_vote_score(single_vote: np.array, set1: set[int], set2: set[int]):

        # Count the agreements between coalition and opposition
        coalition_against = (single_vote[list(set1)] == -1).sum()
        coalition_in_favor = (single_vote[list(set1)] == 1).sum()
        opposition_against = (single_vote[list(set2)] == -1).sum()
        opposition_in_favor = (single_vote[list(set2)] == 1).sum()

        # Calculate the total number of coalition and opposition members who participated in the vote
        total_coalition = coalition_against + coalition_in_favor
        total_opposition = opposition_against + opposition_in_favor
        total_in_favor = coalition_in_favor + opposition_in_favor
        total_against = coalition_against + opposition_against
        total_participation = total_coalition + total_opposition

        # Calculate the level of agreement/disagreement between coalition and opposition

        # Division score: the more agreement, the higher the score
        phi_c_o = abs((coalition_in_favor * opposition_against - coalition_against * opposition_in_favor) / np.sqrt(
            total_in_favor * total_against * total_coalition * total_opposition + 1))

        # Adjust the score by the total number of participants (i.e., 120 members vs fewer participants)
        participation_factor = total_participation / 120
        weighted_phi_c_o = phi_c_o * participation_factor

        return weighted_phi_c_o

    n_mks, n_votes = votes_matrix.shape
    return sum(single_vote_score(votes_matrix[:, i], set1, set2) for i in range(n_votes)) / n_votes


def compute_yuval_score(votes_matrix: np.array, set1: set[int], set2: set[int]) -> float:

    def single_vote_score(single_vote: np.array, set1: set[int], set2: set[int]):
        # Count the agreements between coalition and opposition
        coalition_against = (single_vote[list(set1)] == -1).sum()
        opposition_against = (single_vote[list(set2)] == -1).sum()

        # Calculate the total number of coalition and opposition members who participated in the vote
        total_coalition = (single_vote[list(set1)] != 0).sum() + 1
        total_opposition = (single_vote[list(set2)] != 0).sum()
        total_participation = total_coalition + total_opposition

        if total_opposition == 0 or total_coalition == 0:
            return 0

        # Division score: the more agreement, the higher the score
        yuv_score = (opposition_against / total_opposition - 0.5) * (
                coalition_against / total_coalition - 0.5) * -4

        # Adjust the score by the total number of participants (i.e., 120 members vs fewer participants)
        participation_factor = total_participation / 120
        weighted_phi_c_o = yuv_score * participation_factor
        return weighted_phi_c_o

    n_mks, n_votes = votes_matrix.shape
    return sum(single_vote_score(votes_matrix[:, i], set1, set2) for i in range(n_votes)) / n_votes


def compute_division(
    votes: pd.DataFrame,
    coallition_opposition: pd.DataFrame,
    method: Literal["cosine", "phi", "Yuval"]
) -> float:

    if len(votes) == 0:
        return 0

    method_to_func = {
        "cosine": _compute_cosine_dist,
        "phi": compute_phi_score,
        "Yuval": compute_yuval_score,
    }

    return compute_score(votes, coallition_opposition, method_to_func[method])
