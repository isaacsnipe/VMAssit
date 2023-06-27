from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from core.config import vla_model
from schemas import error, classify

router = APIRouter()


# def check_if_model_exists() -> None:
#     if vla_model.get_model() is None:
#         raise HTTPException(500, "ML model not found")


@router.post(
    "/classify",
    response_model=classify.ClassificationOut,
    responses={
        400: {"model": error.InvalidDocumentError},
        500: {"model": error.MLModelNotFoundError},
    },
)
async def document_classification(document: UploadFile = File(...)):
    """
    Use this API for Classifying legal documents.
    How to use:
    1. Enter document as pdf.
    2. Click execute.
    3. JSON output will be generated with document class.
    Example:
        {
            "Label":"Court case"
        }
    """

    # label = vla_model.predict(vla_model.PDF_FILE_PATH)
    label = vla_model.predict(document, document.filename)
    return label
