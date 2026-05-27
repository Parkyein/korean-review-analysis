import os
from dataclasses import dataclass
from typing import Any, Iterable, TypedDict

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from elasticsearch import Elasticsearch, helpers
except ImportError:
    Elasticsearch = None
    helpers = None


class SearchOutputItem(TypedDict):
    review_id: str
    score: float
    product_name: str | None
    content: str


class SearchOutput(TypedDict):
    query: str
    results: list[SearchOutputItem]


# TODO: Update this mapping to match the documents you will index.
DEFAULT_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "review_id": {"type": "keyword"},
            "product_name": {"type": "keyword"},
            "rating": {"type": "integer"},
            "content": {"type": "text", "analyzer": "standard"},
        }
    }
}


@dataclass
class ElasticsearchConfig:
    url: str
    index_name: str
    verify_certs: bool = True


def load_config(env_path: str | None = None) -> ElasticsearchConfig:
    """Load Elasticsearch settings from .env or environment variables.

    Expected variables:
    - ELASTICSEARCH_URL
    - ELASTICSEARCH_INDEX
    - ELASTICSEARCH_VERIFY_CERTS
    """
    if load_dotenv is not None:
        load_dotenv(env_path)

    return ElasticsearchConfig(
        url=os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"),
        index_name=os.getenv("ELASTICSEARCH_INDEX", "reviews"),
        verify_certs=os.getenv("ELASTICSEARCH_VERIFY_CERTS", "true").lower() == "true",
    )


def get_client(config: ElasticsearchConfig | None = None) -> Elasticsearch:
    """Create an Elasticsearch client from config."""
    if Elasticsearch is None:
        raise ImportError("Install elasticsearch first: pip install elasticsearch")

    config = config or load_config()
    client_options = {
        "hosts": [config.url],
        "verify_certs": config.verify_certs,
    }

    return Elasticsearch(**client_options)


def preprocess_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one raw record into the document schema for indexing."""
    # TODO: Normalize fields, clean text, and add any metadata you need.
    raise NotImplementedError("preprocess_record is not implemented yet.")


def build_documents(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build preprocessed documents before calling index_documents."""
    return [preprocess_record(record) for record in records]


def create_index(
    client: Elasticsearch,
    index_name: str,
    mapping: dict[str, Any] | None = None,
    recreate: bool = False,
) -> None:
    """Create an index with a supplied mapping.

    Set recreate=True to delete and recreate the index.
    """
    mapping = mapping or DEFAULT_INDEX_MAPPING

    if client.indices.exists(index=index_name):
        if not recreate:
            return
        client.indices.delete(index=index_name)

    client.indices.create(index=index_name, **mapping)


def index_documents(
    documents: Iterable[dict[str, Any]],
    client: Elasticsearch | None = None,
    index_name: str | None = None,
    mapping: dict[str, Any] | None = None,
    recreate: bool = False,
) -> tuple[int, list[Any]]:
    """Bulk index documents that are already preprocessed."""
    if helpers is None:
        raise ImportError("Install elasticsearch first: pip install elasticsearch")

    config = load_config()
    client = client or get_client(config)
    index_name = index_name or config.index_name

    create_index(client, index_name=index_name, mapping=mapping, recreate=recreate)

    actions = [
        {
            "_index": index_name,
            "_id": document["review_id"],
            "_source": document,
        }
        for document in documents
    ]

    return helpers.bulk(client, actions, refresh="wait_for")


def build_search_body(query: str, size: int = 10) -> dict[str, Any]:
    """Build an Elasticsearch query body from a user query."""
    # TODO: Implement your retrieval strategy.
    raise NotImplementedError("build_search_body is not implemented yet.")


def postprocess_results(
    results: list[dict[str, Any]],
    query: str,
) -> SearchOutput:
    """
    Post-process search results before returning them.
    (e.g. reranking, filtering, transforming)
    """
    # TODO: Rerank, filter, or transform search results.
    raise NotImplementedError("postprocess_results is not implemented yet.")


def validate_search_output(output: Any) -> bool:
    """Check whether search output follows the required schema."""
    if not isinstance(output, dict):
        return False
    if not isinstance(output.get("query"), str):
        return False
    if not isinstance(output.get("results"), list):
        return False

    for item in output["results"]:
        if not isinstance(item, dict):
            return False
        if not isinstance(item.get("review_id"), str):
            return False
        if not isinstance(item.get("score"), (int, float)):
            return False
        if item.get("product_name") is not None and not isinstance(
            item.get("product_name"),
            str,
        ):
            return False
        if not isinstance(item.get("content"), str):
            return False

    return True


def search(
    query: str,
    client: Elasticsearch | None = None,
    index_name: str | None = None,
    size: int = 10,
) -> SearchOutput:
    """Run the full retrieval pipeline and return validated search output."""
    # TODO: Implement your retrieval pipeline.
    raise NotImplementedError("search is not implemented yet.")
