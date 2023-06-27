import fastapi
import uvicorn


from flair.data import Sentence
from flair.models import SequenceTagger

from pydantic import BaseModel

model = SequenceTagger.load('flair/ner-english-ontonotes-large')

app = fastapi.FastAPI()


class Document(BaseModel):
    body: str

class DocumentResponse(BaseModel):
    classes: list

from typing import Dict, List

@app.post("/")
async def document_classification(document: Document):
    response = {}
    sentence = Sentence(document.body)
    model.predict(sentence, return_probabilities_for_all_classes=True)
    
    taglist = []
    
    for entity in sentence.get_spans('ner'):
        pair = {"tag": entity.tag, "text":entity.text}
        taglist.append(pair)
        
    response["classes"] = taglist
    
    return response


if __name__ == "__main__":
    uvicorn.run(
        "test_main:app",
        host="0.0.0.0",
        port=8083,
        reload=False,
    )
