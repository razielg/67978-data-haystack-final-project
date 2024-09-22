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


def get_all_votes(logger, kvg: KnessetVotesCollector):
    logger.info("Starting to fetch all votes by range, this might take a long time (up to ~4h)")

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


def get_all_mk_ids(logger, kvg: KnessetVotesCollector):
    logger.info("Starting to fetch all MK ids")
    res = kvg.get_all_mk_ids()
    res.to_csv(os.path.join(RESULTS_DIR, f"mk_ids.csv"))
    logger.info("Done")


def get_all_vip_ids(logger, kvg: KnessetVotesCollector):
    logger.info("Starting to fetch all MK VIP ids")
    res = kvg.get_all_vip_ids()
    res.to_csv(os.path.join(RESULTS_DIR, f"vip_ids.csv"))
    logger.info("Done")


def get_all_vote_details(logger, kvg: KnessetVotesCollector):
    logger.info("Starting to fetch all vote details")

    votes_per_chunk = MAX_VOTE_ID
    for i in tqdm.tqdm(range(0, MAX_VOTE_ID, votes_per_chunk)):

        range_min = i
        range_max = i + votes_per_chunk - 1

        try:
            res = kvg.get_vote_details_range(range_min, range_max)
            res.to_csv(os.path.join(RESULTS_DIR, f"vd_{range_min:05d}_to_{range_max:05d}.csv"))
        except OdataQueryException as e:  # don't get stuck if we don't succeed
            logger.warning(f"Got {e} while getting votes {range_min:,}-{range_max:,}, skipping")
            continue

    logger.info("Done")


def create_output_dir():
    try:
        os.mkdir(RESULTS_DIR)
    except FileExistsError:
        pass


def main():

    logger = setup_logger()
    create_output_dir()

    kvg = KnessetVotesCollector(KNESSET_VOTES_SVC, logger)
    get_all_votes(logger, kvg)


if __name__ == '__main__':
    main()
