from fastapi import FastAPI

app = FastAPI()

@app.get("/hello")
def srl():
    return 'Hello, World'