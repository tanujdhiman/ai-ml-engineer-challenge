# **Machine Learning Engineer Coding Challenge**  
## **Personality Type Predictor API**  

### **📌 Overview**  
The goal of this challenge is to **build an API that predicts a user’s MBTI personality type** based on text input.  

This challenge is **not about building the most accurate model** but rather **about demonstrating your ability to set up an end-to-end system**—including data processing, model serving, API development, containerization, and CI/CD.  

---

## **🛠️ Challenge Specifications**  

### **1️⃣ Problem Statement**  
You are tasked with building a **machine learning-powered API** that predicts a person’s MBTI personality type based on their text input. You will:  
- Train an ML model using the **MBTI dataset** ([Kaggle Reddit clean dataset](https://www.kaggle.com/datasets/zeyadkhalid/mbti-personality-types-500-dataset/data)).  
- Deploy the model as a **REST API** (choice of web framework is up to you).  
- Containerize the application using **Docker**.  
- Implement **CI/CD automation** for testing and building the API.  
- **(Bonus)** Integrate a **self-hosted open-source monitoring tool** to track API requests and model confidence scores.  

### **2️⃣ Technical Requirements**  

#### **🧠 Machine Learning**
- Train a simple classifier that predicts an MBTI type.  
- Model accuracy **is not the main focus**, but the system must output a personality type.  
- **(Bonus)** Include a **confidence score** in the API response.  

#### **🖥️ API Development**  
- Implement a REST API with **at least one endpoint**:  
  - `POST /predict` → Accepts a JSON payload with text input and returns the predicted MBTI type (+ optional confidence).  
  - Example Input:  
    ```json
    {"text": "I love deep conversations and thinking about abstract ideas."}
    ```  
  - Example Output:  
    ```json
    {
      "mbti_type": "INFJ",
      "confidence": 0.87
    }
    ```  
- You **choose the web framework** (Django, Flask, FastAPI, etc.).  

#### **📦 Containerization**  
- Provide a **Dockerfile** to package the API and dependencies.  
- The API should run locally using `docker run`.  

#### **🛠️ CI/CD Pipeline**  
- Use **GitHub Actions** (or GitLab CI) to:  
  - Run **automated tests** for API functionality. Keep it simple: one or two simple tests are enough.
  - Build a **Docker image** upon successful tests.  

---

## **✅ Key Takeaways and additional guidelines**  
- The emphasis is on building a **functional end-to-end system**, rather than optimizing the model’s performance. That said, we do expect to see a vectorization + ML model or NLP processing.
- **Keep** the training and model creation code also on this repo.
- Apart from using Python as the programming language, you **may choose your tech stack**, as long as the required functionality is met.
- Feel free to do **AI-assisted programming** for this task, however we expect that you do demonstrate clear understanding and control over the entire process, including every choice made.

