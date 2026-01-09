from fastapi import FastAPI

app = FastAPI(title="AI Server")

@app.get("/health")
async def health():
    return {"status": "ok"}
