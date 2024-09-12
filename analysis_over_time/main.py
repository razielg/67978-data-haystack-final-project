import datetime
from typing import Iterable

import pandas as pd
from matplotlib import pyplot as plt

from data import datasets

from communities_analysis import compute_communities_cosine_score


def compute_division_index(votes: pd.DataFrame) -> float:
    """
    TODO
    Compute the division index for a given votes dataframe.
    :param votes: Vote results table. Assumes all votes are from the same Knesset.
    :return: The division index.
    """
    return compute_communities_cosine_score(votes)


def iter_by_knesset(
        vote_details: pd.DataFrame, vote_results: pd.DataFrame) -> Iterable[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """
    Iterate over given votes (details and per-MK results), yielding a subset of votes from the same Knesset num
    on each iteration. Assumes both input dataframes are sorted with increasing vote_id.
    """
    min_knesset = 16
    max_knesset = 24

    for knesset_num in range(min_knesset, max_knesset + 1):
        yield (
            knesset_num,
            vote_details[vote_details["knesset_num"] == knesset_num],
            vote_results[vote_results["knesset_num"] == knesset_num],
        )


def iter_by_timeframe(
        vote_details: pd.DataFrame, vote_results: pd.DataFrame,
        start: datetime.datetime, end: datetime.datetime, interval: datetime.datetime
) -> Iterable[tuple[datetime.datetime, pd.DataFrame, pd.DataFrame]]:
    """
    TODO
    Iterate over given votes (details and per-MK results), yielding a subset of votes from the same time interval
    on each iteration. Assumes both input dataframes are sorted with increasing vote_id.
    """
    pass


def main():
    all_vote_results = datasets.get_all_votes()
    all_vote_details = datasets.get_all_vote_details()

    # Division index iver Knessets
    division_index_vals = [(knesset, compute_division_index(res)) for (knesset, _, res) in iter_by_knesset(
        all_vote_details, all_vote_results)]
    x, y = zip(*division_index_vals)
    plt.plot(x, y, "ro")
    plt.show()

    # TODO compute monthly division index around 2nd Lebanon war, Protective Edge, and Guardian of the Walls + plot
    pass


if __name__ == '__main__':
    main()
