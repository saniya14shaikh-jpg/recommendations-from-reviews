"""
Test Suite for Recommendations from Reviews API
Run: pytest tests/test_api.py -v
"""

import pytest, json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c

def test_health(client):
    r = client.get('/api/health')
    assert r.status_code == 200
    d = json.loads(r.data)
    assert d['status'] == 'ok'

def test_preprocess(client):
    r = client.post('/api/preprocess',
                    json={'text': 'This is AMAZING!!! https://amazon.com 123'},
                    content_type='application/json')
    assert r.status_code == 200
    d = json.loads(r.data)
    assert 'cleaned' in d
    assert 'tokens' in d

def test_predict_single(client):
    r = client.post('/api/predict',
                    json={'text': 'Great product, works perfectly!', 'model': 'classical'},
                    content_type='application/json')
    assert r.status_code == 200
    d = json.loads(r.data)
    assert 'label' in d
    assert d['label'] in [0, 1]
    assert 'confidence' in d

def test_predict_batch(client):
    texts = ['Amazing product!', 'Terrible quality, broke immediately.', 'Average item.']
    r = client.post('/api/predict/batch',
                    json={'texts': texts, 'model': 'classical'},
                    content_type='application/json')
    assert r.status_code == 200
    d = json.loads(r.data)
    assert d['total'] == 3
    assert len(d['results']) == 3

def test_predict_empty_text(client):
    r = client.post('/api/predict', json={'text': ''}, content_type='application/json')
    assert r.status_code == 400

def test_stats(client):
    r = client.get('/api/stats')
    assert r.status_code == 200
    d = json.loads(r.data)
    assert 'total' in d

def test_history(client):
    r = client.get('/api/history?limit=10')
    assert r.status_code == 200
    assert isinstance(json.loads(r.data), list)