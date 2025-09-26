from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="InsightDesk AI")

class Ticket(BaseModel):
    ticket_id: str
    description: str

@app.get("/")
def read_root():
    return {"message": "InsightDesk AI API is running."}

@app.post("/tickets")
def create_ticket(ticket: Ticket):
    # Placeholder: route ticket to ML pipeline
    return {"ticket_id": ticket.ticket_id, "status": "received"}
