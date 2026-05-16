Student Performance Predictor 🎓

A machine learning web application that predicts student performance based on study hours and attendance percentage using a Linear Regression model. The project is built with Python, Scikit-learn, and Streamlit for an interactive user interface.

The application allows users to input student details and instantly receive a predicted success score along with a pass/fail indication.

Live Demo link : https://linearregression-a06.streamlit.app/

🚀 Features
Predicts student success using Machine Learning
Interactive web interface using Streamlit
Uses Linear Regression for prediction
Real-time prediction updates
Simple and beginner-friendly project
Visualization-ready structure
🛠️ Technologies Used
Python
Streamlit
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
📂 Project Structure
├── app.py                           # Streamlit web application
├── train.py                         # Model training script
├── Student_Result_5000_Dataset.csv # Dataset
├── model.pkl                        # Saved trained model
├── scaler.pkl                       # Saved scaler
└── README.md                        # Project documentation
📊 Dataset

The dataset contains:

Hours_Studied → Number of study hours
Attendance → Attendance percentage
Result → Student performance result
⚙️ Installation
1. Clone the Repository
git clone https://github.com/your-username/student-performance-predictor.git
cd student-performance-predictor
2. Install Dependencies
pip install -r requirements.txt
▶️ Run the Application
streamlit run app.py

The application will open in your browser automatically.

🧠 Model Training

The model is trained using Linear Regression.

Training process includes:

Loading dataset
Splitting data into training and testing sets
Training the model
Evaluating using:
Mean Squared Error (MSE)
R² Score

Training code is available in train.py.

📌 Application Workflow
User enters:
Study Hours
Attendance Percentage
The model predicts student success score
Application displays:
Predicted Success Percentage
Pass/Fail Result

The Streamlit app implementation is available in app.py.

📈 Example Prediction
Hours Studied	Attendance	Prediction
8	90%	Likely Pass ✅
2	40%	Likely Fail ❌
🎯 Future Improvements
Add more student performance features
Use advanced ML algorithms
Deploy using Streamlit Cloud or Heroku
Add graphical analytics dashboard
Improve model accuracy
👨‍💻 Author

Anand

📜 License

This project is open-source and available under the MIT Licens
