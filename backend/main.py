import fastapi
import uvicorn
from fastapi import responses

from api.v1.routers import health, classify
from core.config import vla_model, settings
from core.ner_flair import flair_model
from core.pydoctr import ocrpdf
from core.trocr_model import trocrModel


app = fastapi.FastAPI(
    title=settings.APP_NAME,
    version=settings.releaseId,
)

@app.on_event("startup")
async def startup_event():
    flair_model.initialize_model()
    ocrpdf.initialize_model()
    trocrModel.initialize_model()
    vla_model.initialize_model()

# NOTE: uncomment this when you are running it on local or on prem
# MODEL_CLASSIFICATION_CHECKPOINT = '/home/ghana/VirtualLegalAssistant/VLA_API/core/checkpoint-768'
# MODEL_EXTRACTION_CHECKPOINT = '/home/ghana/VirtualLegalAssistant/VLA_API/core/checkpoint-4608'

app.include_router(health.router, prefix=settings.API_V1_STR, tags=["health"])
app.include_router(classify.router, prefix=settings.API_V1_STR, tags=["classify"])


@app.get("/", include_in_schema=False)
async def index() -> responses.RedirectResponse:
    return responses.RedirectResponse(url="/docs")

# if __name__ == "__main__":
#     # uvicorn.run(app, debug=True)
#     uvicorn.run(app)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8183,
        reload=True,
    )
    

