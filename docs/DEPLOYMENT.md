# 🐳 Docker Hub Deployment Guide

Step-by-step guide to build and deploy to Docker Hub.

---

## 📋 Prerequisites

1. Docker installed ([Download](https://www.docker.com/get-started))
2. Docker Hub account ([Sign up](https://hub.docker.com/signup))
3. Trained model in `models/checkpoints_real/`

---

## 🏗️ Building the Docker Image

### Step 1: Prepare Files
```bash
cd ~/vision-ai-system

# Ensure model exists
ls models/checkpoints_real/best_model.pth

# Optional: Test locally first
streamlit run app.py
```

### Step 2: Build Image
```bash
# Build image (takes 3-5 minutes)
docker build -t vision-ai-system .

# Check image size
docker images | grep vision-ai-system
```

**Expected output:**
```
vision-ai-system   latest   abc123   5 minutes ago   2.1GB
```

### Step 3: Test Locally
```bash
# Run container
docker run -p 8501:8501 vision-ai-system

# Open browser at http://localhost:8501
# Test prediction, then stop (Ctrl+C)
```

---

## 🌐 Pushing to Docker Hub

### Step 1: Login
```bash
# Login to Docker Hub
docker login

# Enter username and password when prompted
```

### Step 2: Tag Image
```bash
# Tag image with your Docker Hub username
docker tag vision-ai-system ahmedyasir779/vision-ai-system:latest

# Optional: Add version tag
docker tag vision-ai-system ahmedyasir779/vision-ai-system:v1.0
```

### Step 3: Push to Hub
```bash
# Push latest tag
docker push ahmedyasir779/vision-ai-system:latest

# Push version tag (optional)
docker push ahmedyasir779/vision-ai-system:v1.0
```

**This takes 5-10 minutes depending on your internet speed.**

### Step 4: Verify on Docker Hub

1. Go to https://hub.docker.com/
2. Navigate to your repository
3. Verify image is uploaded
4. Update description (optional)

---

## 🚀 Deploying from Docker Hub

### Anyone can now run your app:
```bash
# Pull image
docker pull ahmedyasir779/vision-ai-system:latest

# Run container
docker run -p 8501:8501 ahmedyasir779/vision-ai-system:latest

# Access at http://localhost:8501
```

### Using Docker Compose:
```yaml
version: '3.8'

services:
  vision-ai:
    image: ahmedyasir779/vision-ai-system:latest
    ports:
      - "8501:8501"
    restart: unless-stopped
```
```bash
docker-compose up -d
```

---

## 🔧 Update & Versioning

### Update Image
```bash
# Make changes to code
# Rebuild image
docker build -t vision-ai-system .

# Tag new version
docker tag vision-ai-system ahmedyasir779/vision-ai-system:v1.1
docker tag vision-ai-system ahmedyasir779/vision-ai-system:latest

# Push both tags
docker push ahmedyasir779/vision-ai-system:v1.1
docker push ahmedyasir779/vision-ai-system:latest
```

### Version Naming

- `latest`: Most recent stable version
- `v1.0`: Specific version number
- `dev`: Development version

---

## 📊 Docker Hub Best Practices

### 1. Add Description

On Docker Hub repository page:
- Click "Edit"
- Add description from README
- Add usage instructions
- Save changes

### 2. Add README

Link to your GitHub README:
```
https://github.com/ahmedyasir779/vision-ai-system
```

### 3. Make Public

Settings → Visibility → Public

---

## 🌍 Cloud Deployment

### Deploy to Cloud Run (Google Cloud)
```bash
# Tag for Google Container Registry
docker tag vision-ai-system gcr.io/YOUR-PROJECT/vision-ai-system

# Push to GCR
docker push gcr.io/YOUR-PROJECT/vision-ai-system

# Deploy to Cloud Run
gcloud run deploy vision-ai-system \
  --image gcr.io/YOUR-PROJECT/vision-ai-system \
  --platform managed \
  --port 8501
```

### Deploy to AWS ECS

1. Create ECR repository
2. Push image to ECR
3. Create ECS task definition
4. Deploy ECS service

### Deploy to Heroku
```bash
# Login to Heroku Container Registry
heroku container:login

# Tag and push
heroku container:push web -a your-app-name
heroku container:release web -a your-app-name
```

---

## 📈 Monitoring

### View Logs
```bash
# Docker logs
docker logs -f <container_id>

# Docker Compose logs
docker-compose logs -f
```

### Health Check
```bash
# Check container health
docker ps

# Test health endpoint
curl http://localhost:8501/_stcore/health
```

---

## 🛠️ Troubleshooting

### Image too large
- Remove unnecessary files
- Use multi-stage builds
- Optimize dependencies

### Container crashes
```bash
# Check logs
docker logs <container_id>

# Run interactively
docker run -it vision-ai-system /bin/bash
```

### Port already in use
```bash
# Use different port
docker run -p 8502:8501 vision-ai-system
```

---

## 📞 Support

Issues? Check:
- [GitHub Issues](https://github.com/ahmedyasir779/vision-ai-system/issues)
- [Docker Hub](https://hub.docker.com/r/ahmedyasir779/vision-ai-system)