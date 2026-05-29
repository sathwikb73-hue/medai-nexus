"""
MedAI Nexus — Hospitals Routes
GET /nearby · GET /{id}
Uses Google Places API with Redis caching (15 min TTL).
"""
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
import httpx, json, logging

from core.config import settings
from core.redis_client import cache_get, cache_set
from middleware.auth import get_current_user
from models.models import User

router = APIRouter()
logger = logging.getLogger("medai.hospitals")

PLACES_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

# Fallback static data when no Maps key configured
FALLBACK_HOSPITALS = [
    {"id": "h1", "name": "AIIMS Hyderabad",   "distance_km": 2.1, "type": "Government",
     "specialty": "Multi-specialty", "phone": "040-2323-4567", "rating": 4.5,
     "is_emergency": True,  "lat": 17.4485, "lon": 78.3908},
    {"id": "h2", "name": "Apollo Hospitals",   "distance_km": 3.4, "type": "Private",
     "specialty": "Cardiac & Trauma", "phone": "040-2360-7777", "rating": 4.7,
     "is_emergency": True,  "lat": 17.4239, "lon": 78.4738},
    {"id": "h3", "name": "Yashoda Hospital",   "distance_km": 4.2, "type": "Private",
     "specialty": "Emergency Care", "phone": "040-2979-9999", "rating": 4.3,
     "is_emergency": True,  "lat": 17.4504, "lon": 78.3820},
    {"id": "h4", "name": "Care Hospitals",     "distance_km": 5.8, "type": "Private",
     "specialty": "Oncology & Neurology", "phone": "040-3041-8888", "rating": 4.2,
     "is_emergency": False, "lat": 17.4156, "lon": 78.4516},
]


@router.get("/nearby")
async def nearby_hospitals(
    lat: float = Query(..., description="User latitude"),
    lon: float = Query(..., description="User longitude"),
    radius_m: int = Query(10000, le=50000),
    emergency_only: bool = False,
    current_user: User = Depends(get_current_user),
):
    cache_key = f"hospitals:{lat:.3f}:{lon:.3f}:{radius_m}:{emergency_only}"
    cached = await cache_get(cache_key)
    if cached:
        return json.loads(cached)

    if not settings.GOOGLE_MAPS_KEY:
        result = FALLBACK_HOSPITALS
        if emergency_only:
            result = [h for h in result if h["is_emergency"]]
        await cache_set(cache_key, json.dumps(result), ttl=900)
        return result

    try:
        params = {
            "location":  f"{lat},{lon}",
            "radius":    radius_m,
            "type":      "hospital",
            "key":       settings.GOOGLE_MAPS_KEY,
        }
        if emergency_only:
            params["keyword"] = "emergency"

        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(PLACES_URL, params=params)
            data = resp.json()

        results = []
        for p in data.get("results", [])[:12]:
            loc = p.get("geometry", {}).get("location", {})
            results.append({
                "id":          p.get("place_id"),
                "name":        p.get("name"),
                "address":     p.get("vicinity"),
                "lat":         loc.get("lat"),
                "lon":         loc.get("lng"),
                "rating":      p.get("rating"),
                "is_emergency": "emergency" in p.get("name", "").lower(),
            })

        await cache_set(cache_key, json.dumps(results), ttl=900)
        return results

    except Exception as e:
        logger.error(f"[Hospitals] Maps API failed: {e}")
        return FALLBACK_HOSPITALS
