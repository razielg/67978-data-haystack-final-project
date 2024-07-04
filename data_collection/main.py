import os
import tqdm
import logging
from rich.logging import RichHandler

from odata_client import OdataQueryException
from knesset_odata import KnessetVotesCollector


KNESSET_VOTES_SVC = "https://knesset.gov.il/Odata/Votes.svc"
MAX_VOTE_ID = 34_525  # as of 02/07/2024, see https://knesset.gov.il/Odata/Votes.svc/vote_rslts_kmmbr_shadow?$orderby=vote_id%20desc&$select=vote_id&$top=1&$format=json
RESULTS_DIR = "vote_results"


def setup_logger():
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()]
    )

    return logger


def main():

    logger = setup_logger()

    try:
        os.mkdir(RESULTS_DIR)
    except FileExistsError:
        pass

    kvg = KnessetVotesCollector(KNESSET_VOTES_SVC, logger)
    logger.info("Created votes collector, starting to fetch all votes by range")

    votes_per_chunk = 1_000
    for i in tqdm.tqdm(range(0, MAX_VOTE_ID, votes_per_chunk)):

        range_min = i
        range_max = i + votes_per_chunk - 1

        try:
            res = kvg.get_votes_range(range_min, range_max)
            res.to_csv(os.path.join(RESULTS_DIR, f"vr_{range_min:05d}_to_{range_max:05d}.csv"))
        except OdataQueryException as e:  # don't get stuck if we don't succeed
            logger.warning(f"Got {e} while getting votes {range_min:,}-{range_max:,}, skipping")
            continue

    logger.info("Done")


if __name__ == '__main__':
    main()
