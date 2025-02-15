# Personality Type Predictor API
 
*Predict your MBTI personality type based on text input using Machine Learning and FastAPI.*

---

## 📌 Overview

The **Personality Type Predictor API** is a machine learning-powered application that predicts a user's MBTI (Myers-Briggs Type Indicator) personality type based on their text input. This project demonstrates end-to-end development, including data processing, model training, API development, containerization, and CI/CD automation.

---

## 🚀 Features

- **MBTI Personality Prediction**: Predicts one of the 16 MBTI types based on user-provided text.
- **Confidence Score**: Provides a confidence score for the prediction.
- **Modern Frontend**: A clean, interactive, and visually appealing frontend for user interaction.
- **REST API**: Built with **FastAPI** for high performance and scalability.
- **Containerization**: Dockerized for easy deployment and reproducibility.
- **CI/CD Pipeline**: Automated testing and Docker image building using GitHub Actions.
- **Monitoring**: Integrated with **Prometheus** and **Grafana** for API and model performance monitoring.

---

## 🛠️ Technical Stack

- **Machine Learning**: Scikit-learn, Random Forest Classifier, TF-IDF Vectorization.
- **Backend**: FastAPI, Python.
- **Frontend**: HTML, CSS, JavaScript.
- **Containerization**: Docker.
- **CI/CD**: GitHub Actions.
- **Monitoring**: Prometheus, Grafana.

---

## 📂 Project Structure

```
mbti-predictor-api/
├── data/                       # Dataset folder
│   └── MBTI 500.csv            # MBTI dataset
├── models/                     # Trained models
│   └── mbti_model.joblib       # Serialized model
├── static/                     # Static files (CSS, JS)
│   ├── css/
│   │   └── styles.css          # Frontend styles
│   └── js/
│       └── script.js           # Frontend interactivity
├── templates/                  # HTML templates
│   └── index.html              # Frontend page
├── tests/                      # Test cases
│   ├── __init__.py
│   ├── test_api.py             # API tests
│   └── test_model.py           # Model tests
├── media/                      # Images and Recordings
│   ├── home.png                  # Home Image
│   ├── recording.mp4             # Recorded Video
├── app.py                      # FastAPI application
├── train.py                    # Model training script
├── predictor.py                # Model prediction script
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Dockerfile for containerization
├── docker-compose.yml          # Docker Compose for monitoring
├── prometheus.yml              # Prometheus configuration
├── .env                        # Environment variables
├── .github/                    # CI/CD workflows
│   └── workflows/
│       └── ci-cd.yml           # GitHub Actions workflow
└── README.md                   # Project documentation
```

---

## 🧠 Machine Learning Model

### Model Details
- **Algorithm**: Random Forest Classifier.
- **Vectorization**: TF-IDF with n-grams (1, 2).
- **Accuracy**: Achieved ~XX% accuracy on the test set (can be updated after training).

### Training Process
1. **Data Preprocessing**:
   - Load and clean the MBTI dataset.
   - Split the data into training and testing sets.
2. **Model Training**:
   - Train a Random Forest classifier using TF-IDF vectorization.
3. **Model Evaluation**:
   - Evaluate the model on the test set and log accuracy.
4. **Model Serialization**:
   - Save the trained model to `models/mbti_model.joblib`.

---

## 🖥️ API Endpoints

### `POST /predict`
Predicts the MBTI personality type based on the input text.

#### Request
```json
{
  "text": "I love deep conversations and thinking about abstract ideas."
}
```

#### Response
```json
{
  "mbti_type": "INFJ",
  "confidence": 0.87
}
```

---

## 🎨 Frontend

### Features
- **Interactive Input Field**: Users can enter text to get their MBTI prediction.
- **Loading Animation**: Displays a spinner while the prediction is being processed.
- **Attractive Result Display**: Shows the predicted MBTI type and confidence score in a visually appealing way.

### Screenshots

#### Home Page
![Home Page](media/home.png)

#### Demo Video
<iframe width="800" height="450" src="https://youtu.be/Y2DyW1e_veM" frameborder="0" allowfullscreen></iframe>

---

## 🐳 Dockerization

The project is containerized using Docker for easy deployment.

### Build Docker Image
```bash
docker build -t mbti-predictor-api .
```

### Run Docker Container
```bash
docker run -p 8000:8000 mbti-predictor-api
```

---

## 🔧 CI/CD Pipeline

The project uses **GitHub Actions** for continuous integration and deployment.

### Workflow Steps
1. **Run Tests**: Execute unit tests for the API and model.
2. **Build Docker Image**: Build the Docker image upon successful tests.
3. **Push to Docker Hub**: Push the Docker image to Docker Hub (optional).

---

## 📊 Monitoring

The project is integrated with **Prometheus** and **Grafana** for monitoring API requests and model performance.

### Setup
1. Start the monitoring stack:
   ```bash
   docker-compose up
   ```
2. Access Prometheus: `http://localhost:9090`
3. Access Grafana: `http://localhost:3000` (default credentials: admin/admin)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Docker (optional)
- Git

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/tanujdhiman/ai-ml-engineer-challenge.git
   cd ai-ml-engineer-challenge
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python train.py
   ```
4. Run the API:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
5. Access the frontend: `http://localhost:8000`

---

## 🧪 Running Tests

To run the unit tests:
```bash
pytest tests/
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset: [MBTI Kaggle Dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type)
- FastAPI: [FastAPI Documentation](https://fastapi.tiangolo.com/)
- Scikit-learn: [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

Made with ❤️ by **Tanuj**  
🚀 Happy Coding!

---
