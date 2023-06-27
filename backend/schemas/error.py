from pydantic import BaseModel


class MLModelNotFoundError(BaseModel):
    detail: str = "ML model not found"


class InvalidDocumentError(BaseModel):
    detail: str = "Invalid Document Type"
