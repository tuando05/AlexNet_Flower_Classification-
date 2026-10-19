"""Kiểm thử REST API của web app (dùng predictor giả, không load checkpoint)."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import config
from app import main as app_main


class FakePredictor:
    """Predictor giả để test API mà không cần checkpoint ~240MB."""

    def __init__(self, class_names=None, is_trained=True):
        self.class_names = list(class_names or config.TARGET_CLASSES)
        self.class_names_vi = config.CLASS_NAMES_VI
        self.is_trained = is_trained
        self.metadata = {
            'arch': 'alexnet',
            'use_pretrained': True,
            'trained_at': '2026-10-18T22:12:08',
            'epoch': 13,
            'val_acc': 0.966,
            'num_classes': len(self.class_names),
            'image_size': 224,
            'batch_size': 32,
            'learning_rate': 1e-4,
        }

    def predict_image(self, image):
        best = self.class_names[0]
        return {
            'prediction': best,
            'prediction_vi': self.class_names_vi.get(best, best),
            'confidence': 0.9,
            'confidence_percentage': 90.0,
            'model_trained': self.is_trained,
            'probabilities': {
                cls: {
                    'name_en': cls,
                    'name_vi': self.class_names_vi.get(cls, cls),
                    'probability': 0.33,
                    'percentage': 33.0,
                }
                for cls in self.class_names
            },
        }


def make_image_bytes(fmt='JPEG', size=(64, 64)):
    buffer = io.BytesIO()
    Image.new('RGB', size, (120, 30, 200)).save(buffer, fmt)
    return buffer.getvalue()


@pytest.fixture
def client(monkeypatch):
    """Client với lifespan chạy thật nhưng predictor được thay bằng bản giả."""
    monkeypatch.setattr(app_main, 'create_predictor', lambda: FakePredictor())
    with TestClient(app_main.app) as test_client:
        yield test_client


@pytest.fixture
def untrained_client(monkeypatch):
    monkeypatch.setattr(app_main, 'create_predictor', lambda: FakePredictor(is_trained=False))
    with TestClient(app_main.app) as test_client:
        yield test_client


# ---------------------------------------------------------------- health

def test_health_reports_model_metadata(client):
    body = client.get('/api/health').json()

    assert body['status'] == 'online'
    assert body['model_loaded'] is True
    assert body['model_trained'] is True
    assert body['target_classes'] == config.TARGET_CLASSES
    assert body['loaded_classes'] == config.TARGET_CLASSES
    assert body['device'] == str(config.DEVICE)
    assert body['max_upload_mb'] == config.MAX_UPLOAD_SIZE_MB

    model = body['model']
    assert model['arch'] == 'alexnet'
    assert model['use_pretrained'] is True
    assert model['best_epoch'] == 13
    assert model['best_val_acc'] == 0.966
    assert model['trained_at'] == '2026-10-18T22:12:08'


def test_health_flags_untrained_model(untrained_client):
    body = untrained_client.get('/api/health').json()

    assert body['model_loaded'] is True
    assert body['model_trained'] is False


# ---------------------------------------------------------------- trang chủ

def test_root_renders_html(client):
    response = client.get('/')

    assert response.status_code == 200
    assert 'text/html' in response.headers['content-type']
    assert 'AlexNet Flower Classifier' in response.text


def test_root_uses_real_sample_images(client):
    response = client.get('/')

    if not app_main.get_sample_images():
        pytest.skip("Chưa tạo ảnh mẫu (python scripts/make_sample_images.py)")

    assert '/samples/' in response.text, "phải nhúng ảnh mẫu thật"
    assert 'sample-img-placeholder' not in response.text, "không được dùng lại placeholder canvas/emoji"


# ---------------------------------------------------------------- predict

def test_predict_returns_prediction(client):
    response = client.post(
        '/api/predict',
        files={'file': ('hoa.jpg', make_image_bytes(), 'image/jpeg')},
    )

    assert response.status_code == 200
    body = response.json()
    assert body['success'] is True

    data = body['data']
    assert data['prediction'] in config.TARGET_CLASSES
    assert data['prediction_vi'] == config.CLASS_NAMES_VI[data['prediction']]
    assert data['filename'] == 'hoa.jpg'
    assert data['image_size'] == [64, 64]
    assert len(data['probabilities']) == len(config.TARGET_CLASSES)


def test_predict_accepts_png(client):
    response = client.post(
        '/api/predict',
        files={'file': ('hoa.png', make_image_bytes('PNG'), 'image/png')},
    )

    assert response.status_code == 200


def test_predict_rejects_non_image_content_type(client):
    response = client.post(
        '/api/predict',
        files={'file': ('ghichu.txt', b'hello', 'text/plain')},
    )

    assert response.status_code == 400
    assert 'không phải ảnh' in response.json()['detail']


def test_predict_rejects_unsupported_extension(client):
    response = client.post(
        '/api/predict',
        files={'file': ('anh.gif', make_image_bytes('GIF'), 'image/gif')},
    )

    assert response.status_code == 400
    assert '.gif' in response.json()['detail']


def test_predict_rejects_corrupt_image(client):
    response = client.post(
        '/api/predict',
        files={'file': ('hong.jpg', b'day khong phai anh', 'image/jpeg')},
    )

    assert response.status_code == 400
    assert 'Không đọc được nội dung ảnh' in response.json()['detail']


def test_predict_rejects_empty_file(client):
    response = client.post(
        '/api/predict',
        files={'file': ('rong.jpg', b'', 'image/jpeg')},
    )

    assert response.status_code == 400
    assert 'File rỗng' in response.json()['detail']


def test_predict_rejects_oversized_file(client):
    oversized = b'\xff' * (config.MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1)

    response = client.post(
        '/api/predict',
        files={'file': ('to.jpg', oversized, 'image/jpeg')},
    )

    assert response.status_code == 413
    assert f"giới hạn {config.MAX_UPLOAD_SIZE_MB}MB" in response.json()['detail']


def test_predict_marks_untrained_model(untrained_client):
    response = untrained_client.post(
        '/api/predict',
        files={'file': ('hoa.jpg', make_image_bytes(), 'image/jpeg')},
    )

    assert response.status_code == 200
    assert response.json()['data']['model_trained'] is False


# ---------------------------------------------------------------- ảnh mẫu

def test_sample_images_are_served(client):
    samples = app_main.get_sample_images()
    if not samples:
        pytest.skip("Chưa tạo ảnh mẫu (python scripts/make_sample_images.py)")

    for sample in samples:
        response = client.get(sample['url'])
        assert response.status_code == 200
        assert response.headers['content-type'].startswith('image/')
        assert len(response.content) > 0


def test_get_sample_images_shape():
    for sample in app_main.get_sample_images():
        assert set(sample) == {'key', 'label', 'filename', 'url'}
        assert sample['key'] in config.TARGET_CLASSES
        assert sample['label'] == config.CLASS_NAMES_VI[sample['key']]


def test_api_samples_endpoint(client):
    body = client.get('/api/samples').json()

    assert body['success'] is True
    assert body['count'] == len(body['samples'])
    assert body['count'] == len(app_main.get_sample_images())

    for sample in body['samples']:
        assert sample['url'].startswith('/samples/')
