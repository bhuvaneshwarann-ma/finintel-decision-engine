import pytest
import jwt
from datetime import datetime, timezone, timedelta
from httpx import AsyncClient, ASGITransport
from main import app
from auth.database import auth_db
from auth.auth_service import auth_service, get_jwt_secret
from config import JWT_ALGORITHM

@pytest.fixture
def unique_email():
    import uuid
    return f"investor_{uuid.uuid4().hex[:8]}@example.com"

# ---------------------------------------------------------
# Test 1: Successful registration
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_1_successful_registration(unique_email):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/auth/register", json={
            "email": unique_email,
            "password": "StrongPassword123!"
        })
        assert res.status_code == 201
        data = res.json()
        assert data["message"] == "Registration successful"
        assert "id" in data["user"]
        assert data["user"]["email"] == unique_email.lower()
        # Verify no password or hash is exposed
        assert "password" not in data["user"]
        assert "password_hash" not in data["user"]

# ---------------------------------------------------------
# Test 2: Duplicate registration returns HTTP 409
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_2_duplicate_registration_returns_409(unique_email):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # First registration
        await client.post("/auth/register", json={
            "email": unique_email,
            "password": "StrongPassword123!"
        })
        # Duplicate registration
        res2 = await client.post("/auth/register", json={
            "email": unique_email,
            "password": "AnotherPassword456!"
        })
        assert res2.status_code == 409
        assert "already registered" in res2.json()["detail"].lower()

# ---------------------------------------------------------
# Test 3: Weak password (< 8 chars) rejected with HTTP 422
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_3_weak_password_rejected():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/auth/register", json={
            "email": "weak_pw@example.com",
            "password": "short"
        })
        assert res.status_code == 422

# ---------------------------------------------------------
# Test 4: Successful login returns JWT Bearer token
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_4_successful_login(unique_email):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/auth/register", json={
            "email": unique_email,
            "password": "StrongPassword123!"
        })
        res = await client.post("/auth/login", json={
            "email": unique_email,
            "password": "StrongPassword123!"
        })
        assert res.status_code == 200
        data = res.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

# ---------------------------------------------------------
# Test 5: Invalid password returns HTTP 401
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_5_invalid_password_returns_401(unique_email):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/auth/register", json={
            "email": unique_email,
            "password": "CorrectPassword123!"
        })
        res = await client.post("/auth/login", json={
            "email": unique_email,
            "password": "WrongPassword999!"
        })
        assert res.status_code == 401
        assert "invalid email or password" in res.json()["detail"].lower()

# ---------------------------------------------------------
# Test 6: Invalid email format returns HTTP 422
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_6_invalid_email_format():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/auth/login", json={
            "email": "not-an-email",
            "password": "SomePassword123!"
        })
        assert res.status_code == 422

# ---------------------------------------------------------
# Test 7: Missing token on protected endpoint returns HTTP 401
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_7_missing_token_returns_401():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/api/analyze", json={
            "ticker": "TATAMOTORS",
            "persona": "conservative"
        })
        assert res.status_code == 401

# ---------------------------------------------------------
# Test 8: Invalid token format/signature returns HTTP 401
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_8_invalid_token_returns_401():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/api/analyze", json={
            "ticker": "TATAMOTORS",
            "persona": "conservative"
        }, headers={"Authorization": "Bearer invalid.token.payload"})
        assert res.status_code == 401

# ---------------------------------------------------------
# Test 9: Expired token returns HTTP 401
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_9_expired_token_returns_401(unique_email):
    secret = get_jwt_secret()
    expired_payload = {
        "sub": "usr_expired123",
        "email": unique_email,
        "iat": int((datetime.now(timezone.utc) - timedelta(hours=2)).timestamp()),
        "exp": int((datetime.now(timezone.utc) - timedelta(hours=1)).timestamp())
    }
    expired_token = jwt.encode(expired_payload, secret, algorithm=JWT_ALGORITHM)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/auth/me", headers={"Authorization": f"Bearer {expired_token}"})
        assert res.status_code == 401
        assert "expired" in res.json()["detail"].lower()

# ---------------------------------------------------------
# Test 10: /auth/me returns authenticated user identity
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_10_auth_me_returns_identity(unique_email):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post("/auth/register", json={"email": unique_email, "password": "StrongPassword123!"})
        login = await client.post("/auth/login", json={"email": unique_email, "password": "StrongPassword123!"})
        token = login.json()["access_token"]

        me_res = await client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me_res.status_code == 200
        me = me_res.json()
        assert me["email"] == unique_email.lower()
        assert me["is_active"] is True
        assert "password" not in me
        assert "password_hash" not in me

# ---------------------------------------------------------
# Test 11: Protected analyze endpoint succeeds with valid token
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_11_analyze_endpoint_with_auth(unique_email):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/auth/register", json={"email": unique_email, "password": "StrongPassword123!"})
        login = await client.post("/auth/login", json={"email": unique_email, "password": "StrongPassword123!"})
        token = login.json()["access_token"]

        res = await client.post("/api/analyze", json={
            "ticker": "TATAMOTORS",
            "persona": "conservative"
        }, headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        assert res.json()["ticker"] == "TATAMOTORS"

# ---------------------------------------------------------
# Test 12: User Profile isolation (GET and PUT /api/profile)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_12_user_profile_crud_and_isolation(unique_email):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/auth/register", json={"email": unique_email, "password": "StrongPassword123!"})
        login = await client.post("/auth/login", json={"email": unique_email, "password": "StrongPassword123!"})
        token = login.json()["access_token"]

        # 1. Default profile
        prof_res = await client.get("/api/profile", headers={"Authorization": f"Bearer {token}"})
        assert prof_res.status_code == 200
        assert prof_res.json()["risk_profile"] == "conservative"

        # 2. Update profile
        update_res = await client.put("/api/profile", json={
            "risk_profile": "aggressive",
            "portfolio_concentration": 0.25
        }, headers={"Authorization": f"Bearer {token}"})
        assert update_res.status_code == 200
        assert update_res.json()["risk_profile"] == "aggressive"
        assert update_res.json()["portfolio_concentration"] == 0.25

# ---------------------------------------------------------
# Test 13: Multi-User Thesis Isolation (Alice vs Bob)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_13_multi_user_thesis_isolation():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        alice_email = "alice_thesis@example.com"
        bob_email = "bob_thesis@example.com"

        # Register & Login Alice
        await client.post("/auth/register", json={"email": alice_email, "password": "AlicePassword123!"})
        alice_login = await client.post("/auth/login", json={"email": alice_email, "password": "AlicePassword123!"})
        alice_token = alice_login.json()["access_token"]

        # Register & Login Bob
        await client.post("/auth/register", json={"email": bob_email, "password": "BobPassword123!"})
        bob_login = await client.post("/auth/login", json={"email": bob_email, "password": "BobPassword123!"})
        bob_token = bob_login.json()["access_token"]

        # Alice creates a private thesis for XYZ_CORP
        alice_th_res = await client.post("/api/thesis", json={
            "ticker": "XYZ_CORP",
            "stated_reasons": ["Alice's secret green hydrogen catalyst"],
            "key_assumptions": ["debt stays below 1.5"],
            "invalidating_conditions": ["debt surge"]
        }, headers={"Authorization": f"Bearer {alice_token}"})
        assert alice_th_res.status_code == 200

        # Alice can view her thesis
        alice_view = await client.get("/api/thesis/XYZ_CORP", headers={"Authorization": f"Bearer {alice_token}"})
        assert alice_view.status_code == 200
        assert alice_view.json()["thesis"] is not None
        assert "Alice's secret" in alice_view.json()["thesis"]["stated_reasons"][0]

        # Bob queries XYZ_CORP thesis -> must NOT see Alice's thesis!
        bob_view = await client.get("/api/thesis/XYZ_CORP", headers={"Authorization": f"Bearer {bob_token}"})
        assert bob_view.status_code == 200
        assert bob_view.json()["thesis"] is None

# ---------------------------------------------------------
# Test 14: Multi-User Session Isolation (Alice vs Bob)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_14_multi_user_session_isolation():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        alice_email = "alice_session@example.com"
        bob_email = "bob_session@example.com"

        # Register & Login Alice
        await client.post("/auth/register", json={"email": alice_email, "password": "AlicePassword123!"})
        alice_login = await client.post("/auth/login", json={"email": alice_email, "password": "AlicePassword123!"})
        alice_token = alice_login.json()["access_token"]

        # Register & Login Bob
        await client.post("/auth/register", json={"email": bob_email, "password": "BobPassword123!"})
        bob_login = await client.post("/auth/login", json={"email": bob_email, "password": "BobPassword123!"})
        bob_token = bob_login.json()["access_token"]

        # Alice runs an analysis -> creates session
        alice_analysis = await client.post("/api/analyze", json={
            "ticker": "INFOSYS",
            "persona": "conservative"
        }, headers={"Authorization": f"Bearer {alice_token}"})
        assert alice_analysis.status_code == 200
        session_id = alice_analysis.json()["session_id"]

        # Alice accesses her evidence graph
        alice_graph = await client.get(f"/api/evidence-graph/{session_id}", headers={"Authorization": f"Bearer {alice_token}"})
        assert alice_graph.status_code == 200

        # Bob tries to access Alice's evidence graph -> HTTP 404
        bob_graph = await client.get(f"/api/evidence-graph/{session_id}", headers={"Authorization": f"Bearer {bob_token}"})
        assert bob_graph.status_code == 404

# ---------------------------------------------------------
# Test 15: Brute Force Rate Limiting (§24)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_15_brute_force_rate_limiting():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        target_email = "brute_target@example.com"
        await client.post("/auth/register", json={"email": target_email, "password": "RealPassword123!"})

        # 5 failed attempts
        for _ in range(5):
            res = await client.post("/auth/login", json={"email": target_email, "password": "WrongPassword!"})
            assert res.status_code == 401

        # 6th attempt should be rate limited with HTTP 429
        res_locked = await client.post("/auth/login", json={"email": target_email, "password": "RealPassword123!"})
        assert res_locked.status_code == 429
        assert "too many failed" in res_locked.json()["detail"].lower()

# ---------------------------------------------------------
# Test 16: Security Headers Presence (§22)
# ---------------------------------------------------------
@pytest.mark.asyncio
async def test_16_security_headers():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/health")
        assert res.status_code == 200
        assert res.headers.get("X-Content-Type-Options") == "nosniff"
        assert res.headers.get("X-Frame-Options") == "DENY"
        assert res.headers.get("Referrer-Policy") == "no-referrer"
