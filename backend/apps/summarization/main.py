from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import logging
from pydantic import BaseModel
from typing import List

from utils import (
    extract_text,
    extract_tables,
    process_kpi_tables,
    get_summary_from_ollama,
)
from utils.utils import get_current_user
from config import SRC_LOG_LEVELS, AppConfig

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["SUMMARIZATION"])

app = FastAPI()

app.state.config = AppConfig()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SummaryResponse(BaseModel):
    summary: str
    kpiTable: List[dict]


@app.get("/")
async def get_status():
    return {"status": "Summarization service is running"}


@app.post("/summarize", response_model=SummaryResponse)
async def summarize(file: UploadFile = File(...), user=Depends(get_current_user)):
    try:
        document_text = extract_text(file)
        tables = extract_tables(file)

        summary = get_summary_from_ollama(document_text)
        kpi_table = process_kpi_tables(tables)

        return SummaryResponse(summary=summary, kpiTable=kpi_table)
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while summarizing the document.",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
