import time
import requests
import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class SingleQueryResult:
    result: pd.DataFrame  # parsed result
    next_required_query: str | None  # specifies what additional query is required to gather the full answer, if at all


class OdataQueryException(Exception):
    pass


@dataclass
class QueryRequestRetryLaterException(OdataQueryException):
    cooldown: int  # seconds we should wait before next request


class QueryRequestFailedException(OdataQueryException):
    pass


class ResponseParser(ABC):
    @abstractmethod
    def parse(self, response_status: int, response_content: bytes) -> SingleQueryResult:
        raise NotImplementedError()


class OdataQuerier:
    def __init__(self, svc_url: str, parser: ResponseParser, logger):
        self.svc_url = svc_url
        self.parser = parser
        self.logger = logger

    def _do_request(self, query: str) -> tuple[int, bytes]:
        """
        Do an HTTP request to the service
        :param query: odata query string
        :return: HTTP status code, HTTP content
        """
        req_url = f"{self.svc_url}/{query}"
        response = requests.get(req_url)
        return response.status_code, response.content

    def _do_single_query(self, query: str) -> SingleQueryResult:
        return self.parser.parse(*self._do_request(query))

    def do_query(self, query: str):
        query_result = self._do_single_query(query)
        res_list = [query_result.result]
        while query_result.next_required_query is not None:
            try:
                query_result = self._do_single_query(query_result.next_required_query)
            except QueryRequestRetryLaterException as e:
                self.logger.debug(f"Got {type(e).__name__} on query #{len(res_list)}, waiting for {e.cooldown}s")
                time.sleep(e.cooldown)
                continue

            res_list.append(query_result.result)

        return pd.concat(res_list)
