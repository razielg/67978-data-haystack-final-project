import datetime
from typing import Iterable

import pandas as pd
from matplotlib import pyplot as plt

from data import datasets

from division_score import compute_division


def compute_division_index(votes: pd.DataFrame, coallition_opposition: pd.DataFrame) -> float:
    """
    Compute the division index for a given votes dataframe.
    :param votes: Vote results table. Assumes all votes are from the same Knesset.
    :return: The division index.
    """
    return compute_division(votes, coallition_opposition, method="phi")


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
        start: datetime.datetime, end: datetime.datetime, interval: datetime.timedelta,
        allow_multiple_knessets: bool = False,
) -> Iterable[tuple[datetime.datetime, pd.DataFrame, pd.DataFrame]]:
    """
    Iterate over given votes (details and per-MK results), yielding a subset of votes from the same time interval
    on each iteration. Assumes both input dataframes are sorted with increasing vote_id.
    """
    df_time_format = "%Y-%m-%dT00:00:00"
    vote_details = vote_details[vote_details["vote_date"] < end.strftime(df_time_format)]
    while start <= end:
        curr_end = start + interval
        vote_details = vote_details[vote_details["vote_date"] >= start.strftime(df_time_format)]
        relevant_vote_details = vote_details[vote_details["vote_date"] < curr_end.strftime(df_time_format)]

        vote_ids = relevant_vote_details["vote_id"].unique()
        relevant_vote_results = vote_results[vote_results["vote_id"].isin(vote_ids)]

        knessets = relevant_vote_details["knesset_num"].unique()
        if len(knessets) == 0:
            yield start, relevant_vote_details, relevant_vote_results, 23  # 23 is just a default
        elif len(knessets) == 1:
            yield start, relevant_vote_details, relevant_vote_results, knessets[0]
        elif not allow_multiple_knessets:
            raise ValueError(f"The timeframe {start}-{curr_end} spans across multiple Knessets {knessets}")
        else:
            for knesset in knessets:
                vd = relevant_vote_details[relevant_vote_details["knesset_num"] == knesset]
                vr = relevant_vote_results[relevant_vote_results["vote_id"].isin(relevant_vote_details["vote_id"].unique())]
                yield start, vd, vr, knesset
        start += interval


def plot_division(
        division_index_vals: list[tuple[int | str, float]],
        title: str, xlabel: str, ylabel: str = "Division Index",
        xticks: list[str] = None,
        occasions: list[tuple[int, str]] = None,
        width=2,
    ):
    plt.rc("axes", axisbelow=True)
    plt.grid()
    x, y = zip(*division_index_vals)
    plt.bar([_x + width / 2 for _x in x], y, width=width, color="tab:blue", edgecolor="black", linewidth=.5)
    plt.ylim(0, .6)
    plt.title(title, size=20)
    plt.xlabel(xlabel, size=15)
    plt.ylabel(ylabel, size=15)

    if xticks:
        resolution = 1 if len(x) < 10 else 2
        plt.xticks(size=15, labels=xticks[::resolution], ticks=x[::resolution], rotation=-25)
    else:
        plt.xticks(size=15)

    plt.yticks(size=15)

    if occasions:
        ytext_off = [0, -.15, -.3]
        y_off = [0, -.15, -.10]
        for i, (occ_x, occ_label) in enumerate(occasions):
            plt.axvline(x=occ_x, color="black", linestyle='--')
            plt.annotate(
                occ_label,
                xy=(occ_x, .42 + y_off[i]),
                xytext=(occ_x - 2.5, .5 + ytext_off[i]),
                bbox=dict(facecolor="white", alpha=0.8, linewidth=.5),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=14,
                color='black',
            )

    plt.tight_layout()
    plt.savefig(f"division_{title.lower().replace(' ', '_')}.png")
    plt.show()


def plot_around_occasion_monthly(
        start: datetime.datetime, end: datetime.datetime,
        vote_details: pd.DataFrame, vote_results: pd.DataFrame,
        coalition_opposition_by_knesset: dict[int, pd.DataFrame],
        title: str, xlabel:str = "Month", ylabel: str = "Division Index",
        occasions: list[tuple[float, str]] = None,
):
    xticks = list()
    interval = datetime.timedelta(days=61)
    curr_tick = start
    while curr_tick <= end:
        xticks.append(curr_tick.strftime("%m/%Y"))
        curr_tick += interval

    index_vals = [
        (int((date - start).days / 30) + start.month, compute_division_index(res, coalition_opposition_by_knesset[knesset])) for (
        date, _, res, knesset) in iter_by_timeframe(vote_details, vote_results, start, end, interval)
    ]
    plot_division(index_vals, title=title, xlabel=xlabel, ylabel=ylabel, xticks=xticks[:len(index_vals)], occasions=occasions)


def main():
    all_vote_results = datasets.get_all_votes()
    all_vote_details = datasets.get_all_vote_details()
    coalition_opposition_by_knesset = {kn: datasets.get_coallition_opposition(knesset_num=kn) for kn in range(16, 24)}

    # Division index over Knessets
    division_index_vals = [(knesset, compute_division_index(res, datasets.get_coallition_opposition(knesset))) for (
        knesset, _, res) in iter_by_knesset(all_vote_details, all_vote_results)]
    plot_division(division_index_vals, title="", xlabel="Knesset", width=1)

    lebanon2_start = datetime.datetime(2006, 5, 1)  # 12 / 07
    lebanon2_end = datetime.datetime(2006, 12, 1)  # 14 / 08
    plot_around_occasion_monthly(
        lebanon2_start, lebanon2_end, all_vote_details, all_vote_results,
        coalition_opposition_by_knesset,
        title="Second Lebanon War",
        occasions=[(7 + 12 / 31, "War breaks out"), (8 + 14 / 30, "Ceasefire starts")]
    )

    protective_edge_start = datetime.datetime(2014, 3, 1)
    protective_edge_end = datetime.datetime(2014, 12, 1)
    plot_around_occasion_monthly(
        protective_edge_start, protective_edge_end, all_vote_details, all_vote_results,
        coalition_opposition_by_knesset,
        title="Operation Protective Edge",
        occasions=[
            (6 + 12 / 30, 'Operation "Shuvu Ahim"'),
            (7 + 17 / 31, "Ground invasion begins"),
            (8 + 26 / 30, "Effective ceasefire"),
        ]
    )

    covid_start = datetime.datetime(2020, 4, 1)
    covid_end = datetime.datetime(2020, 11, 1)
    plot_around_occasion_monthly(
        covid_start, covid_end, all_vote_details, all_vote_results,
        coalition_opposition_by_knesset,
        title="COVID Pandemic",
        occasions=[
            (3 + 20 / 31, "First Lockdown"),
            (5 + 3 / 31, "Schools Reopen"),
            (9 + 18 / 30, "Second Lockdown"),
        ]
    )

    subprime_start = datetime.datetime(2008, 5, 1)
    subprime_end = datetime.datetime(2008, 12, 31)
    plot_around_occasion_monthly(
        subprime_start, subprime_end, all_vote_details, all_vote_results,
        coalition_opposition_by_knesset,
        title="Subprime Crisis",
        occasions=[
            (9 + 15 / 30, "Lehman Brothers defuncts"),
            (12 + 25 / 31, "Tel Aviv 100 index\nreaches lowest value"),
        ]
    )


if __name__ == '__main__':
    main()
