from urllib.parse import quote

from fastapi.testclient import TestClient


def test_upload_then_ask_uses_the_uploaded_data(monkeypatch, tmp_path):
    import querypilot.uploads as up
    monkeypatch.setattr(up, "UPLOAD_DIR", tmp_path)
    from querypilot import server
    monkeypatch.setattr(server, "path_for", lambda i: up.path_for(i, tmp_path))
    monkeypatch.setattr(server, "build_database", lambda d, f: up.build_database(d, f, tmp_path))
    client = TestClient(server.app)
    r = client.post("/api/upload", content=b"city;sales\nParis;3\nLyon;4\n",
                    headers={"X-Filename": quote("ventes été.csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["file"] == "ventes été.csv" and body["tables"][0]["name"] == "ventes_ete"
    assert body["examples"]
    r = client.post("/api/ask", json={"question": "anything", "db_id": body["db_id"]})
    assert r.status_code == 200
    assert "ventes_ete" in r.text          # the agent explored the uploaded table
    assert client.post("/api/ask", json={"question": "q", "db_id": "f" * 32}).status_code == 404
    assert client.post("/api/upload", content=b"x", headers={"X-Filename": "a.pdf"}).status_code == 400
