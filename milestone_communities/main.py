import pandas as pd
import numpy as np
import networkx as nx
import itertools

from data.datasets import get_all_votes, VoteResultType, get_all_vip_ids
from visualization import plot_graph_with_color


def pivot_results(vote_results: pd.DataFrame, allowed_vote_values: set, min_votes: int) -> np.array:
    """
    Pivots a dataframe with a row for each results to a matrix `result[kmmbr_id, vote_id] == vote_result`,
    where vote_result is -1 if `kmmbr_id` did not participate in vote `vote_id`.
    Ignores rows with vote results not in `allowed_vote_values`, or rows describing vote_ids that appear less than
    `min_votes` times.
    :return: A tuple of the pivot result as a numpy array, and a list translating this matrix index to the original
    kmmbr_id
    """
    vote_results = vote_results[vote_results["vote_result"].isin(allowed_vote_values)]

    is_vote_big_enough = vote_results["vote_id"].value_counts() >= min_votes
    big_enough_votes = is_vote_big_enough[is_vote_big_enough].index
    vote_results = vote_results[vote_results["vote_id"].isin(big_enough_votes)]

    pivot = vote_results.pivot_table(  # TODO assumes unique rows
        index="kmmbr_id", columns="vote_id", values="vote_result", aggfunc="sum", fill_value=-1)

    return pivot.to_numpy(), list(pivot.index)


def compute_connections(tristate_vote_matrix: np.array) -> np.array:
    """
    Creates a weighted graph representing the strength of connection between every pair of MKs, based on
    their given voting statistics.
    :param tristate_vote_matrix: Matrix, `tristate_vote_matrix[kmmbr_id, vote_id]` is `1` if voted for, `-1` if voted
                                 against, or `0` otherwise.
    :return: Adjacency matrix with weight between every pair of MKs.
    """
    n_mks, n_votes = tristate_vote_matrix.shape

    # each MK contributes a total weight of 1, regardless of how many votes they participated in
    weights_per_mk_vote = 1 / np.abs(tristate_vote_matrix).sum(axis=1)  # shape (n_mks,)

    weighted_agreement_matrix = np.zeros((n_mks, n_mks))
    for mk1, mk2 in itertools.product(range(n_mks), repeat=2):
        # TODO we don't count disagreements

        if mk1 == mk2:
            continue

        weighted_agreement_matrix[mk1, mk2] = \
            ((tristate_vote_matrix[mk1, :] * tristate_vote_matrix[mk2, :]) == 1).sum() \
            * (weights_per_mk_vote[mk1] + weights_per_mk_vote[mk2])

    return weighted_agreement_matrix


def create_connection_graph(
        connections_matrix: np.array, vip_ids: list[int], vip_ids_and_names_dataframe: pd.DataFrame) -> nx.Graph:
    """
    :param connections_matrix: Adjacency matrix, as returned by `create_connection_graph()`
    :param vip_ids: Translation from matrix index to kmmbr_id.
    :return: Input as networkx Graph()
    """
    n, n_ = connections_matrix.shape
    assert n == n_ and n == len(vip_ids)

    df = vip_ids_and_names_dataframe
    df["full_name"] = df.apply(
        lambda row: f"{row['mk_individual_first_name_eng']} {row['mk_individual_name_eng']}", axis=1)
    filtered_df = df[df["vip_id"].isin(vip_ids)]
    full_names = filtered_df.set_index("vip_id").loc[vip_ids]["full_name"].tolist()

    nodes = [(vip_id, {"name": full_name}) for vip_id, full_name in zip(vip_ids, full_names)]

    G: nx.Graph = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(
        ((vip_ids[i], vip_ids[j], connections_matrix[i, j]) for i, j in itertools.product(
            range(n), repeat=2) if i < j))

    return G


def detect_communities(G: nx.Graph) -> list[set[int]]:
    """
    :param G: Graph representation of MKs
    :return: Louvain communities, see networkx's `community.louvain_communities()`
    """
    return nx.community.louvain_communities(G)


def main():
    all_votes: pd.DataFrame = get_all_votes()

    # for boundaries of 23rd Knesset, see following:
    # https://knesset.gov.il/Odata/Votes.svc/vote_rslts_kmmbr_shadow?$orderby=vote_id&$format=json&$select=vote_id&$top=1&$filter=knesset_num%20eq%2023 (32034)
    # https://knesset.gov.il/Odata/Votes.svc/vote_rslts_kmmbr_shadow?$orderby=vote_id%20desc&$format=json&$select=vote_id&$top=1&$filter=knesset_num%20eq%2023 (34042)
    twenty_third_knesset_vote_ids = range(32_034, 34_042 + 1)
    twenty_third_knesset_votes = all_votes[all_votes["vote_id"].isin(twenty_third_knesset_vote_ids)]

    # swap to MK x MK aggregation. `kmmbr_id`s are lost but can be easily reconstructed.
    pivot, idx_to_vip_id = pivot_results(
        twenty_third_knesset_votes,
        allowed_vote_values={VoteResultType.FOR.value, VoteResultType.AGAINST.value},
        min_votes=15,
    )

    # +/-1 for support/oppose, 0 otherwise
    as_tristate: np.array = np.zeros_like(pivot, dtype=int) + (pivot == VoteResultType.FOR.value) - (
            pivot == VoteResultType.AGAINST.value)
    connection_matrix = compute_connections(as_tristate)

    # list of index sets representing the communities
    vip_ids_and_names_dataframe = get_all_vip_ids()
    G: nx.Graph = create_connection_graph(connection_matrix, idx_to_vip_id, vip_ids_and_names_dataframe)
    communities: list[set[int]] = detect_communities(G)

    plot_graph_with_color(connection_matrix, communities)

    # results for 23rd Knesset, compared to https://he.wikipedia.org/wiki/הכנסת_העשרים_ושלוש
    # community #1: 72 members, matches government
    # community #2: 48 members, matches opposition
    # community #3: 22 members, I'm not sure, maybe the Norwegian guys?
    # A total of 142 = 120 + 22 nodes
    print(f"{len(communities) = }")
    print(*[f"{i = } | {len(c) = }" for i, c in enumerate(communities)], sep="\n")

    return 0


if __name__ == '__main__':
    main()
