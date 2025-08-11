from fastapi import FastAPI

app = FastAPI(title="AA-660 Doc AI")

@app.get("/health")
def health():
    return {"status": "ok"}
