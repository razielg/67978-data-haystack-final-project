import json
import pandas as pd

from odata_client import (ResponseParser, QueryRequestFailedException, QueryRequestRetryLaterException,
                          SingleQueryResult, OdataQuerier)


class KnessetResponseParser(ResponseParser):

    @staticmethod
    def _check_status(response_status: int):
        if response_status == 503:
            raise QueryRequestRetryLaterException(61)  # just a guess that worked
        elif response_status != 200:
            raise QueryRequestFailedException(
                f"Query request failed with code {response_status}")

    @staticmethod
    def _parse_good_response(response_content: bytes) -> SingleQueryResult:
        as_dict = json.loads(response_content)
        next_required_query = as_dict.get("odata.nextLink")
        if next_required_query is not None:
            next_required_query = f"{next_required_query}" "&$format=json"  # TODO move format elsewhere

        result = pd.DataFrame(as_dict["value"])
        return SingleQueryResult(result=result, next_required_query=next_required_query)

    def parse(self, response_status: int, response_content: bytes) -> SingleQueryResult:
        self._check_status(response_status)
        return self._parse_good_response(response_content)


class KnessetVotesCollector:
    """
    Client for Knesset's Votes.svc
    """
    def __init__(self, svc_url: str, logger):
        self.svc_url = svc_url
        self.querier = OdataQuerier(svc_url=svc_url, parser=KnessetResponseParser(), logger=logger)
        self.logger = logger

    def get_all_mk_ids(self):
        return self.querier.do_query(
            "View_Vote_MK_Individual"
            "?$select=mk_individual_id,mk_individual_name_eng,mk_individual_first_name_eng"
            "&$orderby=mk_individual_id"
            "&$format=json"
        )

    def get_all_vip_ids(self):
        # This is the same ID provided in `vote_rslts_kmmbr_shadow`
        return self.querier.do_query(
            "View_Vote_MK_Individual"
            "?$select=vip_id,mk_individual_id,mk_individual_name_eng,mk_individual_first_name_eng"
            "&$orderby=vip_id"
            "&$format=json"
        )

    def get_all_vote_ids(self):
        # TODO
        pass

    def get_all_votes(self):
        # max vote_id as of 02/07/2024: 34,525
        # TODO this should take a lot of time, implement some kind of saving mechanism before trying to run
        return self.querier.do_query("vote_rslts_kmmbr_shadow?$orderby=vote_id&$format=json&$select=vote_id,kmmbr_id,vote_result")

    def get_votes_range(self, min_vote_id: int, max_vote_id: int) -> pd.DataFrame:
        return self.querier.do_query(
            "vote_rslts_kmmbr_shadow"
            f"?$filter=vote_id lt {max_vote_id} and vote_id ge {min_vote_id}"
            "&$orderby=vote_id"
            "&$select=vote_id,kmmbr_id,vote_result"
            "&$format=json"
        )

    def get_vote_details_range(self, min_vote_id: int, max_vote_id: int):
        return self.querier.do_query(
            "View_vote_rslts_hdr_Approved"
            f"?$filter=vote_id lt {max_vote_id} and vote_id ge {min_vote_id}"
            "&$orderby=vote_id"
            "&$format=json"
        )
