# MLOps Thực Hành

Một dự án MLOps hoàn chỉnh với YOLO model training, experiment tracking, và CI/CD pipeline tự động.

## 📋 Mô tả

Dự án này triển khai một pipeline MLOps đầy đủ cho việc training và deploy YOLO model **phát hiện đối tượng (Object Detection)** trên dataset COCO128 với 80 loại đối tượng (người, xe cộ, động vật, đồ vật,...), bao gồm:

- **Model Training**: Training YOLO model với Ultralytics
- **Experiment Tracking**: Theo dõi experiments với MLflow
- **Model Registry**: Quản lý và promote models với MLflow Model Registry
- **Data Versioning**: Quản lý data với DVC (Data Version Control)
- **CI/CD Pipeline**: Tự động train, build, và deploy với GitHub Actions
- **API Inference**: FastAPI service để serve model predictions

## 🏗️ Kiến trúc

```
┌─────────────┐
│  GitHub     │
│  (Trigger)  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  GitHub Actions CI/CD Pipeline       │
│  ┌──────────┐  ┌──────────┐         │
│  │ Job 1:   │→ │ Job 2:   │         │
│  │ Train    │  │ Build &  │         │
│  │ on EC2   │  │ Push     │         │
│  └──────────┘  └────┬─────┘         │
│                    │                │
│                    ▼                │
│              ┌──────────┐           │
│              │ Job 3:   │           │
│              │ Deploy   │           │
│              │ to EC2   │           │
│              └──────────┘           │
└─────────────────────────────────────┘
       │                    │
       ▼                    ▼
┌─────────────┐      ┌─────────────┐
│  EC2 Server │      │  GHCR       │
│  (Training) │      │  (Registry) │
└─────────────┘      └─────────────┘
       │
       ▼
┌─────────────┐
│  EC2 Server │
│  (Deploy)   │
│  FastAPI    │
└─────────────┘
```

## 🚀 Tính năng

### 1. Model Training
- Training YOLO models (YOLOv8, YOLOv11) với Ultralytics
- Tự động evaluate trên validation và test sets
- Model promotion logic dựa trên mAP metrics

### 2. Experiment Tracking
- MLflow integration cho experiment tracking
- Log metrics, parameters, và artifacts
- Model versioning và registry

### 3. Data Versioning
- DVC để quản lý datasets
- Tự động detect data changes và trigger retraining
- S3 backend cho data storage

### 4. CI/CD Pipeline
- **Job 1 (train_on_server)**: SSH vào EC2, pull code, train model
- **Job 2 (build_and_push)**: Build Docker image và push lên GHCR
- **Job 3 (deploy_ec2)**: Deploy container lên EC2 server

### 5. API Inference
- FastAPI service với `/predict` endpoint
- Health check endpoint
- Model loading từ S3 hoặc local

## 📦 Cài đặt

### Yêu cầu

- Python 3.11+
- Docker (cho deployment)
- AWS Account (cho S3 storage)
- EC2 instance (cho training và deployment)
- GitHub repository với Actions enabled

### Setup Local

1. **Clone repository**:
```bash
git clone https://github.com/your-username/mlops-thuc-hanh.git
cd mlops-thuc-hanh
```

2. **Cài đặt dependencies**:
```bash
pip install -r requirements.txt
```

3. **Setup DVC**:
```bash
dvc pull
```

4. **Cấu hình environment variables**:
Tạo file `.env`:
```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET=your_bucket_name
MLFLOW_TRACKING_URI=http://localhost:5001
```

5. **Start MLflow server** (optional, cho local tracking):
```bash
mlflow server --backend-store-uri sqlite:///./mlflow.db --serve-artifacts --host 0.0.0.0 --port 5001
```

## 🎯 Sử dụng

### Training Model

```bash
python train.py --epochs 100 --model yolo11n --batch 16
```

Hoặc sử dụng DVC pipeline:
```bash
dvc repro
```

### Chạy API Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Test API

**Local:**
```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

**Production (EC2):**
```bash
# Health check
curl http://13.212.160.80:8000/health

# Root endpoint
curl http://13.212.160.80:8000/

# Prediction
curl -X POST "http://13.212.160.80:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

**Lưu ý:** Đảm bảo Security Group của EC2 instance đã mở port 8000 (TCP) cho inbound traffic.

## 🔧 Cấu hình

### DVC Configuration

File `dvc.yaml` định nghĩa training pipeline. Cấu hình parameters trong `params.yaml`:

```yaml
train:
  epochs: 100
  model: yolo11n
  batch_size: 16
  imgsz: 640
```

### GitHub Actions Secrets

Cần setup các secrets sau trong GitHub repository:

- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `S3_BUCKET`: S3 bucket name
- `EC2_HOST`: EC2 server IP/hostname
- `EC2_USER`: SSH username
- `EC2_KEY`: SSH private key
- `EC2_PORT_SSH`: SSH port (default: 22)
- `EC2_PORT_DEPLOY`: Deploy SSH port (default: 22)
- `GITHUB_TOKEN`: Auto-provided by GitHub Actions

## 📁 Cấu trúc Project

```
mlops-thuc-hanh/
├── app/                    # FastAPI application
│   ├── __init__.py
│   └── main.py            # API endpoints
├── .github/
│   └── workflows/
│       ├── deploy.yml     # CI/CD pipeline
│       └── test-ci.yml    # Test pipeline
├── mlflow/                 # MLflow server config
│   ├── Dockerfile
│   └── docker-compose.yaml
├── scripts/                # Utility scripts
│   ├── hooks/
│   ├── setup_dvc.sh
│   └── trigger_jenkins.py
├── tests/                  # Unit tests
│   ├── __init__.py
│   └── test_main.py
├── data.dvc               # DVC data tracking
├── dvc.yaml               # DVC pipeline definition
├── params.yaml            # Training parameters
├── train.py               # Training script
├── Dockerfile             # Production Docker image
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project metadata
└── README.md              # This file
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest tests/ --cov=app --cov-report=html
```

## 📊 MLflow Dashboard

Sau khi start MLflow server, truy cập:
```
http://localhost:5001
```

Xem experiments, metrics, và model registry.

## 🔄 CI/CD Workflow

Workflow được trigger khi:
- `data/**` files thay đổi
- `data.dvc` file thay đổi
- Manual trigger via `workflow_dispatch`

Pipeline flow:
1. **Train**: SSH vào EC2, train model với DVC
2. **Build**: Build Docker image và push lên GHCR
3. **Deploy**: Deploy container lên EC2

## 🤝 Đóng góp

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

tandat88963820@gmail.com

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO models
- [MLflow](https://mlflow.org/) for experiment tracking
- [DVC](https://dvc.org/) for data versioning
- [FastAPI](https://fastapi.tiangolo.com/) for API framework
