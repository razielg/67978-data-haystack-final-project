import datetime
from typing import Iterable

import pandas as pd
from matplotlib import pyplot as plt

from data import datasets

from communities_analysis import compute_communities_cosine_score


def compute_division_index(votes: pd.DataFrame, coallition_opposition: pd.DataFrame) -> float:
    """
    TODO
    Compute the division index for a given votes dataframe.
    :param votes: Vote results table. Assumes all votes are from the same Knesset.
    :return: The division index.
    """
    return compute_communities_cosine_score(votes, coallition_opposition)


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
        start: datetime.datetime, end: datetime.datetime, interval: datetime.timedelta
) -> Iterable[tuple[datetime.datetime, pd.DataFrame, pd.DataFrame]]:
    """
    TODO
    Iterate over given votes (details and per-MK results), yielding a subset of votes from the same time interval
    on each iteration. Assumes both input dataframes are sorted with increasing vote_id.
    """
    df_time_format = "%Y-%m-%dT00:00:00"
    vote_details = vote_details[vote_details["vote_date"] < end.strftime(df_time_format)]
    while start <= end:
        vote_details = vote_details[vote_details["vote_date"] >= start.strftime(df_time_format)]
        relevant_vote_details = vote_details[vote_details["vote_date"] < end.strftime(df_time_format)]

        vote_ids = relevant_vote_details["vote_id"].unique()
        relevant_vote_results = vote_results[vote_results["vote_id"].isin(vote_ids)]

        knessets = relevant_vote_details["knesset_num"].unique()
        if len(knessets) != 1:
            raise ValueError(f"The timeframe {start}-{end} spans across multiple Knessets {knessets}")

        yield start, relevant_vote_details, relevant_vote_results, knessets[0]
        start += interval


def plot_division(
        division_index_vals: list[tuple[int | str, float]],
        title: str, xlabel: str, ylabel: str = "Division Index",
    ):
    x, y = zip(*division_index_vals)
    plt.plot(x, y, "ro")
    plt.title(title, size=20)
    plt.xlabel(xlabel, size=15)
    plt.ylabel(ylabel, size=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"division_{title.lower().replace(' ', '_')}.png")
    plt.show()


def plot_around_occasion(
        start: datetime.datetime, end: datetime.datetime,
        vote_details: pd.DataFrame, vote_results: pd.DataFrame,
        coalition_opposition_by_knesset: dict[int, pd.DataFrame],
        title: str, xlabel:str = "Month", ylabel: str = "Division Index",
        interval: datetime.timedelta = datetime.timedelta(days=31)):
    index_vals = [(date.month, compute_division_index(res, coalition_opposition_by_knesset[knesset])) for (
        date, _, res, knesset) in iter_by_timeframe(vote_details, vote_results, start, end, interval)]
    plot_division(index_vals, title=title, xlabel=xlabel, ylabel=ylabel)


def main():
    all_vote_results = datasets.get_all_votes()
    all_vote_details = datasets.get_all_vote_details()
    coalition_opposition_by_knesset = {kn: datasets.get_coallition_opposition(knesset_num=kn) for kn in range(16, 24)}

    # Division index iver Knessets
    # TODO add coalition / opposition column and pass it to compute_division_index()
    division_index_vals = [(knesset, compute_division_index(res, datasets.get_coallition_opposition(knesset))) for (
        knesset, _, res) in iter_by_knesset(all_vote_details, all_vote_results)]
    plot_division(division_index_vals, title="Division Index along the Years", xlabel="Knesset")

    # TODO compute monthly division index around 2nd Lebanon war, Protective Edge, and Guardian of the Walls + plot
    monthly_interval = datetime.timedelta(days=31)  # TODO...

    lebanon2_start = datetime.datetime(2006, 5, 1)  # 12 / 07
    lebanon2_end = datetime.datetime(2006, 11, 1)  # 14 / 08
    plot_around_occasion(
        lebanon2_start, lebanon2_end, all_vote_details, all_vote_results,
        coalition_opposition_by_knesset,
        title="Second Lebanon War",
    )

    protective_edge_start = datetime.datetime(2014, 5, 1)
    protective_edge_end = datetime.datetime(2014, 12, 1)
    plot_around_occasion(
        protective_edge_start, protective_edge_end, all_vote_details, all_vote_results,
        coalition_opposition_by_knesset,
        title="Operation Protective Edge",
    )

    disengagement_start = datetime.datetime(2005, 1, 1)
    disengagement_end = datetime.datetime(2005, 12, 1)
    plot_around_occasion(
        disengagement_start, disengagement_end, all_vote_details, all_vote_results,
        coalition_opposition_by_knesset,
        title="The Disengagement from Gaza Strip",
    )


if __name__ == '__main__':
    main()
