# FastAPI APP for IP-LINK

from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
import httpx
import os

app = FastAPI()

# --------- External API Endpoints ---------
GRANTS_SEARCH_URL = "https://api.training.grants.gov/v1/api/search2"
GRANTS_DETAIL_URL = "https://api.training.grants.gov/v1/api/fetchOpportunity"

SAM_SEARCH_URL = "https://api.sam.gov/opportunities/v2/search"
SAM_DETAIL_URL = "https://api.sam.gov/opportunities/v2/{opportunity_id}"
SAM_HISTORY_URL = "https://api.sam.gov/opportunities/v2/{opportunity_id}/history"

SAM_API_KEY = "biLgXMFUdZeTfkwcq5TUyge6Pfpn9RWed2sq7eFA"

# --------- Request Models ---------

class SearchRequest(BaseModel):
    keyword: str
    rows: int = 10
    oppStatuses: str = "forecasted|posted"

class OpportunityDetailRequest(BaseModel):
    opportunityId: int

class ContractSearchRequest(BaseModel):
    keyword: str
    postedFrom: str  # format: MM/DD/YYYY
    postedTo: str    # format: MM/DD/YYYY
    noticeType: str = "solicitation"
    limit: int = 5

# --------- Grants Endpoints ---------

@app.post("/grants/search")
async def search_grants(payload: SearchRequest):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(GRANTS_SEARCH_URL, json=payload.dict())
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/grants/opportunity")
async def fetch_grant_details(payload: OpportunityDetailRequest):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(GRANTS_DETAIL_URL, json=payload.dict())
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------- Contracts Endpoints ---------

@app.post("/contracts/search")
async def search_contracts(payload: ContractSearchRequest):
    params = {
        "api_key": SAM_API_KEY,
        "keywords": payload.keyword,
        "postedFrom": payload.postedFrom,
        "postedTo": payload.postedTo,
        "ptype": payload.noticeType,
        "limit": payload.limit,
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(SAM_SEARCH_URL, params=params)
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/contracts/{opportunity_id}")
async def get_contract_details(opportunity_id: str = Path(...)):
    url = SAM_DETAIL_URL.format(opportunity_id=opportunity_id)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params={"api_key": SAM_API_KEY})
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/contracts/{opportunity_id}/history")
async def get_contract_history(opportunity_id: str = Path(...)):
    url = SAM_HISTORY_URL.format(opportunity_id=opportunity_id)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params={"api_key": SAM_API_KEY})
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
