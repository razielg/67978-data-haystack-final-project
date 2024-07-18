import pandas as pd
import os
from enum import Enum

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")


def __raw_data_path(file_name: str) -> str:
    return os.path.join(RAW_DATA_DIR, file_name)


def get_all_votes() -> pd.DataFrame:
    return pd.read_csv(__raw_data_path("all_vote_results.csv"))


class VoteResultType(Enum):
    # see https://knesset.gov.il/Odata/Votes.svc/vote_result_type?$format=json
    CANCELLED = 0
    FOR = 1
    AGAINST = 2
    ABSTAIN = 3
    DID_NOT_VOTE = 4
