# 🏋️ Workout Type Recommendation System

A machine learning-based system that recommends workout types (Cardio, Strength, Yoga, or HIIT) based on user physical attributes and fitness metrics.

## 📋 Table of Contents

- [Problem Description](#problem-description)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Model Performance](#model-performance)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)

---

## 🎯 Problem Description

### The Challenge

Recommending appropriate workout types based on user physical characteristics and fitness levels. The goal is to build a machine learning model that can predict which type of workout (Cardio, Strength, Yoga, or HIIT) would be most suitable for a person based on their:

- Physical attributes (age, weight, height, BMI)
- Fitness metrics (heart rate, fat percentage)
- Workout behavior (frequency, duration, calories burned)
- Experience level (Beginner, Intermediate, Expert)

### Business Value

- **For Gym Apps**: Personalized workout recommendations
- **For Trainers**: Quick assessment tool for new clients
- **For Users**: Data-driven guidance on workout selection

### ML Approach

This is a **multi-class classification problem** with 4 target classes:
- Cardio
- Strength
- Yoga
- HIIT

---

## 📊 Dataset

**Source**: [Kaggle - Gym Members Exercise Dataset](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset)

**Size**: 973 records, 15 features

### Features:

**Demographics:**
- Age (years)
- Gender (Male/Female)
- Weight (kg)
- Height (m)
- BMI

**Fitness Metrics:**
- Max BPM
- Average BPM
- Resting BPM
- Fat Percentage

**Workout Behavior:**
- Session Duration (hours)
- Calories Burned
- Workout Frequency (days/week)
- Water Intake (liters)

**Experience:**
- Experience Level (Beginner/Intermediate/Expert)

**Target Variable:**
- Workout Type (Cardio/Strength/Yoga/HIIT)

### Data Quality:
- ✅ No missing values
- ✅ Balanced classes (~25% each)
- ✅ Clean data with consistent formats

---

## 📁 Project Structure

```
workout-recommendation/
│
├── data/
│   └── gym_members_exercise_tracking.csv    # Raw dataset
│
├── models/
│   ├── data_prepared.pkl                    # Preprocessed data
│   └── final_model.pkl                      # Trained XGBoost model
│
├── notebook.ipynb                           # Jupyter notebook with EDA
├── train.py                                 # Model training script
├── predict.py                               # Flask API for predictions
├── requirements.txt                         # Python dependencies
├── Dockerfile                               # Docker configuration
├── .dockerignore                            # Docker ignore file
├── .gitignore                               # Git ignore file
└── README.md                                # Project documentation
```

---

## 🔧 Installation

### Prerequisites

- Python 3.9+
- pip
- Docker (optional, for containerization)

### Option 1: Local Setup with Virtual Environment

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/workout-recommendation.git
cd workout-recommendation

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Docker Setup

```bash
# Build Docker image
docker build -t workout-recommendation .

# Run Docker container
docker run -p 5000:5000 workout-recommendation
```

---

## 🚀 Running the Project

### 1. Data Exploration (Optional)

```bash
# Open Jupyter notebook
jupyter notebook notebook.ipynb
```

The notebook contains:
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training and evaluation
- Visualizations

### 2. Train the Model

```bash
# Train the model
python train.py
```

This will:
- Load preprocessed data
- Train XGBoost classifier
- Evaluate on test set
- Save model to `models/final_model.pkl`

### 3. Run the API Server

```bash
# Start Flask server
python predict.py
```

The API will be available at: `http://localhost:5000`

### 4. Test the API

Open your browser and go to `http://localhost:5000` to see the web interface.

**Or use cURL:**

```bash
# Health check
curl http://localhost:5000/health

# Make prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 25,
    "gender": "Male",
    "weight_kg": 70,
    "height_m": 1.75,
    "max_bpm": 180,
    "avg_bpm": 140,
    "resting_bpm": 70,
    "session_duration_hours": 1.0,
    "calories_burned": 400,
    "fat_percentage": 20,
    "water_intake_liters": 2.5,
    "workout_frequency_days_week": 4,
    "experience_level": "Intermediate"
  }'
```

---

## 📈 Model Performance

### Models Compared

| Model | Training Accuracy | Validation Accuracy | Test Accuracy |
|-------|------------------|---------------------|---------------|
| Logistic Regression | 37.4% | 29.2% | - |
| Decision Tree | 50.9% | 28.7% | - |
| Random Forest | 96.2% | 27.7% | - |
| **XGBoost** | **100.0%** | **31.3%** | **20.6%** |

**Selected Model**: XGBoost Classifier

### Performance Analysis

#### Why Is Accuracy Low?

The model achieved modest accuracy (~30% validation, ~21% test) for several important reasons:

**1. The Nature of the Problem**

Workout type preference is primarily driven by **personal choice and personality**, not physical attributes. The dataset contains physical and fitness metrics, but workout preference depends more on:
- Personal interests and goals
- Personality traits (preference for social vs solo activities)
- Past experience and familiarity
- Mental state and mood

**Example**: Two people with identical physical stats (same age, BMI, fitness level) can have completely different workout preferences.

**2. Evidence from the Data**

Analysis shows that physical attributes have minimal variation across workout types:
- Average BMI: ~23.5 across all workout types
- Average age: ~38 across all workout types
- Gender distribution: Balanced across all types

**3. Model Behavior Indicates Lack of Predictive Signal**

The massive overfitting gap (68-69%) in complex models indicates they're fitting to noise rather than real patterns:
- Random Forest: 96.2% train, 27.7% validation (68.5% gap)
- XGBoost: 100% train, 31.3% validation (68.7% gap)

**4. Comparison to Random Baseline**

- Random guessing: 25% accuracy (1 out of 4 classes)
- Our best model: 31% validation accuracy
- **Only 6% improvement over random** - indicating weak predictive power

#### Is This a Failure?

**No! This is an important learning outcome.**

This project successfully demonstrates:
- ✅ Complete ML pipeline from data exploration to deployment
- ✅ Proper methodology (train/validation/test splits, cross-validation)
- ✅ Model comparison and selection techniques
- ✅ Overfitting detection and analysis
- ✅ Critical thinking - recognizing when a problem isn't suitable for ML
- ✅ Production-ready deployment with Docker
- ✅ **Real-world experience**: Not all business problems are solvable with ML

### Key Insights

**What we learned:**
1. Physical attributes don't strongly predict workout type preferences
2. Personal preference problems may not be suitable for traditional ML
3. High training accuracy with low validation accuracy = overfitting to noise
4. Honest reporting of limitations is more valuable than inflating metrics

**What would improve this:**
- Features capturing personality traits, goals, and preferences
- User workout history and feedback
- Different problem formulation (recommendation system vs classification)

---

## 🌐 API Documentation

### Base URL

**Local**: `http://localhost:5000`  
**Production**: `https://YOUR-APP.up.railway.app`

### Endpoints

#### 1. Home Page
```
GET /
```
Returns HTML interface for testing the API.

#### 2. Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "XGBoost",
  "classes": ["Cardio", "HIIT", "Strength", "Yoga"]
}
```

#### 3. Make Prediction
```
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "age": 25,
  "gender": "Male",
  "weight_kg": 70.0,
  "height_m": 1.75,
  "max_bpm": 180,
  "avg_bpm": 140,
  "resting_bpm": 70,
  "session_duration_hours": 1.0,
  "calories_burned": 400.0,
  "fat_percentage": 20.0,
  "water_intake_liters": 2.5,
  "workout_frequency_days_week": 4,
  "experience_level": "Intermediate"
}
```

**Response:**
```json
{
  "workout_type": "HIIT",
  "confidence": 0.35,
  "probabilities": {
    "Cardio": 0.25,
    "HIIT": 0.35,
    "Strength": 0.22,
    "Yoga": 0.18
  }
}
```

**Field Descriptions:**

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| age | integer | 18-100 | User's age in years |
| gender | string | "Male", "Female" | User's gender |
| weight_kg | float | 40-150 | Weight in kilograms |
| height_m | float | 1.4-2.2 | Height in meters |
| max_bpm | integer | 120-220 | Maximum heart rate (BPM) |
| avg_bpm | integer | 80-180 | Average heart rate during exercise |
| resting_bpm | integer | 40-100 | Resting heart rate |
| session_duration_hours | float | 0.5-3.0 | Typical workout duration |
| calories_burned | float | 100-1000 | Calories burned per session |
| fat_percentage | float | 5-50 | Body fat percentage |
| water_intake_liters | float | 1-5 | Daily water intake |
| workout_frequency_days_week | integer | 1-7 | Workouts per week |
| experience_level | string | "Beginner", "Intermediate", "Expert" | Fitness experience level |

---

## 🚀 Deployment

### Local Deployment with Docker

```bash
# Build the Docker image
docker build -t workout-recommendation .

# Run the container
docker run -p 5000:5000 workout-recommendation

# Access the application
open http://localhost:5000
```

### Cloud Deployment (Railway)

The application is deployed on Railway and accessible at:

**🌐 Live URL**: `https://YOUR-APP.up.railway.app`

**Deployment Steps:**

1. Connect GitHub repository to Railway
2. Railway auto-detects Dockerfile
3. Automatic build and deployment
4. Generate public domain

**Note**: Railway free tier provides $5 credit for 30 days, sufficient for this project.

### Environment Variables

No environment variables required for basic functionality. All configurations are in the code.

---

## 🛠️ Technologies Used

### Machine Learning
- **scikit-learn** (1.3.0) - ML algorithms and preprocessing
- **XGBoost** (2.0.0) - Gradient boosting model
- **pandas** (2.1.0) - Data manipulation
- **numpy** (1.24.3) - Numerical operations

### Web Framework
- **Flask** (3.0.0) - Web API
- **gunicorn** - WSGI server (for production)

### Visualization
- **matplotlib** (3.7.2) - Plotting
- **seaborn** (0.12.2) - Statistical visualizations

### Deployment
- **Docker** - Containerization
- **Railway** - Cloud hosting

### Development Tools
- **Jupyter** - Interactive development
- **Git** - Version control

---

## 🔮 Future Improvements

### Model Enhancements
1. **Feature Engineering**
   - Add personality trait indicators
   - Include user goals (weight loss, muscle gain, flexibility)
   - Incorporate past workout history
   - Add social preference indicators

2. **Different Approach**
   - Change from classification to recommendation system
   - Implement collaborative filtering
   - Use clustering to find similar user groups
   - Build a hybrid model combining multiple approaches

3. **More Data**
   - Collect larger dataset with more diverse features
   - Include user feedback and satisfaction ratings
   - Add temporal patterns (workout preferences over time)

### Technical Improvements
1. **API Enhancements**
   - Add authentication
   - Implement rate limiting
   - Add batch prediction endpoint
   - Create user profiles and history tracking

2. **Model Monitoring**
   - Track prediction accuracy over time
   - Implement A/B testing
   - Add model retraining pipeline
   - Monitor data drift

3. **User Interface**
   - Build React/Vue.js frontend
   - Add visualization of user's fitness profile
   - Show comparison with similar users
   - Provide workout plans based on recommendations

---

## 📝 License

This project is for educational purposes as part of a Machine Learning course.

---

## 👤 Author

**Your Name**
- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/your-profile)

---

## 🙏 Acknowledgments

- Dataset from [Kaggle - Gym Members Exercise Dataset](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset)
- ML Zoomcamp course for project structure and guidance
- Railway for free hosting credits

---

## 📞 Contact

For questions or feedback, please open an issue on GitHub or contact me at [your-email@example.com]