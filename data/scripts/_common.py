import logging
import re
from pathlib import Path

from dotenv import load_dotenv
from langfuse import Langfuse, get_client
from langfuse.api.commons.errors.not_found_error import NotFoundError

load_dotenv()

DATA_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = DATA_ROOT / "corpus"
QUESTIONS_ROOT = DATA_ROOT / "questions"

CORPUS_FILENAME_PATTERN = re.compile(
    r"^(?P<subject>[a-z]+)_(?P<grade>\d+)(?:_(?P<semester>\d+))?_ch(?P<chapter>\d+)\.txt$"
)


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("seed")


def get_langfuse_client() -> Langfuse:
    return get_client()


def dataset_exists(client: Langfuse, name: str) -> bool:
    try:
        client.get_dataset(name)
    except NotFoundError:
        return False
    return True


def parse_corpus_filename(filename: str) -> dict | None:
    match = CORPUS_FILENAME_PATTERN.match(filename)
    if not match:
        return None
    return {
        "subject": match.group("subject"),
        "grade": match.group("grade"),
        "semester": match.group("semester"),
        "chapter": int(match.group("chapter")),
    }


def corpus_dataset_name(subject: str, grade: str, semester: str | None) -> str:
    if semester:
        return f"corpus-{subject}-{grade}-{semester}"
    return f"corpus-{subject}-{grade}"
