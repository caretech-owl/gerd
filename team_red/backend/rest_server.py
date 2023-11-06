import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from team_red.config import CONFIG


class QAQuestion(BaseModel):
    query: str


class QAAnswer(BaseModel):
    status: int
    error_msg: str = ""
    answer: str = ""


app = FastAPI()


@app.post("/api/v1/qa/query", status_code=200, response_model=QAAnswer)
def qa_query(*, query: QAQuestion) -> dict:
    return QAAnswer(status=200, answer="42")


if __name__ == "__main__":
    uvicorn.run(
        "team_red.backend:app",
        host=CONFIG.backend.ip,
        port=CONFIG.backend.port,
        reload=True,
    )
