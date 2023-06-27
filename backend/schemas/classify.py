from typing import List

from pydantic import BaseModel


class ClassificationOut(BaseModel):
    file: dict
    label: dict
    contract_end_date: dict
    parties: dict
    site_id: dict
    contract_start_date: dict
