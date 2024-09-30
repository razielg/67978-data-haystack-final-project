import pandas as pd
import os
from enum import Enum

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
RAW_DATA_DIR_MK_BY = os.path.join(RAW_DATA_DIR, "mk_by_specific_knesset")


def __raw_data_path(file_name: str) -> str:
    return os.path.join(RAW_DATA_DIR, file_name)


def __raw_data_path_mk_by_specific_knesset(file_name: str) -> str:
    return os.path.join(RAW_DATA_DIR_MK_BY, file_name)


def get_all_votes() -> pd.DataFrame:
    return pd.read_csv(__raw_data_path("all_vote_results.csv"))


def get_all_mk_ids() -> pd.DataFrame:
    return pd.read_csv(__raw_data_path("all_mk_ids.csv"))


def get_all_vip_ids() -> pd.DataFrame:
    # This is the same ID provided in `get_all_votes()`
    return pd.read_csv(__raw_data_path("all_vip_ids.csv"))


def get_knesset23_is_coalition() -> pd.DataFrame:
    return pd.read_csv(__raw_data_path("knesset23_coalition.csv"))


def get_knesset_is_coalition_by_number(knesset_num: int) -> pd.DataFrame:
    # return pd.read_csv(__raw_data_path(f"merged_mk_by_knesset_{knesset_num}.csv"), encoding='windows-1255')
    return pd.read_csv(__raw_data_path_mk_by_specific_knesset(
        f"merged_mk_by_knesset_{knesset_num}.csv"), encoding='windows-1255')


def get_all_vote_details() -> pd.DataFrame:
    return pd.read_csv(__raw_data_path("all_vote_details.csv"))


class VoteResultType(Enum):
    # see https://knesset.gov.il/Odata/Votes.svc/vote_result_type?$format=json
    CANCELLED = 0
    FOR = 1
    AGAINST = 2
    ABSTAIN = 3
    DID_NOT_VOTE = 4
