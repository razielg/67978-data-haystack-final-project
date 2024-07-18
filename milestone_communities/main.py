import pandas as pd
import numpy as np
import itertools

from data.datasets import get_all_votes, VoteResultType


def pivot_results(vote_results: pd.DataFrame, allowed_vote_values: set, min_votes: int) -> np.array:
    """
    Pivots a dataframe with a row for each results to a matrix `result[kmmbr_id, vote_id] == vote_result`,
    where vote_result is -1 if `kmmbr_id` did not participate in vote `vote_id`.
    Ignores rows with vote results not in `allowed_vote_values`, or rows describing vote_ids that appear less than
    `min_votes` times.
    """
    vote_results = vote_results[vote_results["vote_result"].isin(allowed_vote_values)]

    is_vote_big_enough = vote_results["vote_id"].value_counts() >= min_votes
    big_enough_votes = is_vote_big_enough[is_vote_big_enough].index
    vote_results = vote_results[vote_results["vote_id"].isin(big_enough_votes)]

    pivot = vote_results.pivot_table(  # TODO assumes unique rows
        index="kmmbr_id", columns="vote_id", values="vote_result", aggfunc="sum", fill_value=-1)

    return pivot.to_numpy()


def create_connection_graph(tristate_vote_matrix: np.array) -> np.array:
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
        weighted_agreement_matrix[mk1, mk2] = \
            ((tristate_vote_matrix[mk1, :] * tristate_vote_matrix[mk2, :]) == 1).sum() \
            * (weights_per_mk_vote[mk1] + weights_per_mk_vote[mk2])

    return weighted_agreement_matrix


def main():
    all_votes = get_all_votes()
    pivot = pivot_results(
        all_votes,
        allowed_vote_values={VoteResultType.FOR.value, VoteResultType.AGAINST.value},
        min_votes=15,
    )

    as_tristate = np.zeros_like(pivot, dtype=int) + (pivot == VoteResultType.FOR.value) - (
            pivot == VoteResultType.AGAINST.value)
    conns = create_connection_graph(as_tristate)

    return 0


if __name__ == '__main__':
    main()
