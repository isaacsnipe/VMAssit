from enum import Enum

from pydantic import BaseModel


class Status(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


class Health(BaseModel):
    status: Status
    version: str
    releaseId: str
