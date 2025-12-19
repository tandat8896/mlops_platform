from fastapi.testclient import TestClient
from app.main import app, ml_models
from ultralytics import YOLO
import io
from PIL import Image
import pytest

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def load_model():
    """Load model before running tests"""
    ml_models["yolo11n"] = YOLO('yolo11n.pt')
    yield
    ml_models.clear()


def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["status"] == "running"


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_model_info():
    """Test model info endpoint"""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "classes" in data
    assert "num_classes" in data


def create_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (640, 480), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr


def test_predict_endpoint():
    """Test prediction endpoint with test image"""
    test_image = create_test_image()
    
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", test_image, "image/jpeg")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "filename" in data
    assert "detections" in data
    assert "count" in data
    assert isinstance(data["detections"], list)


def test_predict_without_file():
    """Test prediction endpoint without file"""
    response = client.post("/predict")
    assert response.status_code == 422  # Unprocessable Entity
