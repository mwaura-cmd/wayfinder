import sys
from fastapi.testclient import TestClient

sys.stdout.reconfigure(encoding="utf-8")

from main import app
import config

client = TestClient(app)

print("=" * 60)
print("WAYFINDER AUTHENTICATION & GUEST ACCESS VERIFICATION")
print("=" * 60)

# 1. Test GET /history without auth header (Guest mode)
res_hist = client.get("/history")
print(f"\n[1/3] GET /history (Unauthenticated / Guest): HTTP {res_hist.status_code}")
assert res_hist.status_code == 200, f"Expected 200, got {res_hist.status_code}"

# 2. Test POST /research without auth header (Guest mode)
res_res = client.post("/research", json={"question": "Test guest query", "level": "standard"})
print(f"[2/3] POST /research (Unauthenticated / Guest): HTTP {res_res.status_code}")
assert res_res.status_code == 200, f"Expected 200, got {res_res.status_code}"
data = res_res.json()
assert "task_id" in data, "Missing task_id in response"
print(f"      Task ID generated: {data['task_id']}")

# 3. Test WAYFINDER_API_KEY enforcement when set
config.WAYFINDER_API_KEY = "test_secret_key_123"
try:
    res_blocked = client.get("/history")
    print(f"[3/3] GET /history with WAYFINDER_API_KEY set (No token): HTTP {res_blocked.status_code}")
    assert res_blocked.status_code == 401, f"Expected 401, got {res_blocked.status_code}"

    res_auth = client.get("/history", headers={"Authorization": "Bearer test_secret_key_123"})
    print(f"      GET /history with valid Bearer token: HTTP {res_auth.status_code}")
    assert res_auth.status_code == 200, f"Expected 200, got {res_auth.status_code}"
finally:
    config.WAYFINDER_API_KEY = ""

print("\n" + "=" * 60)
print("ALL AUTHENTICATION & GUEST ACCESS VERIFICATIONS PASSED!")
print("=" * 60)
