from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')
import random
import re

# Initialize Flask app
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)  # needed for url_for to generate with https
app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', 'dev-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///' + os.path.join(app.instance_path, 'student_prediction.db'))
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_recycle': 300,
    'pool_pre_ping': True,
} if os.environ.get('DATABASE_URL') else {}
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize extensions
db = SQLAlchemy()
db.init_app(app) 
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='student')  # student, faculty, admin
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    class_name = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(120))
    phone = db.Column(db.String(20))
    guardian_email = db.Column(db.String(120))
    guardian_phone = db.Column(db.String(20))
    current_risk_level = db.Column(db.String(20), default='safe')  # safe, warning, high_risk
    risk_score = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
class AttendanceRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(20), db.ForeignKey('student.student_id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    status = db.Column(db.String(10), nullable=False)  # present, absent, late
    subject = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class TestScore(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(20), db.ForeignKey('student.student_id'), nullable=False)
    subject = db.Column(db.String(50), nullable=False)
    test_name = db.Column(db.String(100), nullable=False)
    score = db.Column(db.Float, nullable=False)
    max_score = db.Column(db.Float, nullable=False)
    test_date = db.Column(db.Date, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Faculty(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    faculty_id = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20))
    department = db.Column(db.String(50))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Assignment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    subject = db.Column(db.String(50), nullable=False)
    class_name = db.Column(db.String(20), nullable=False)
    due_date = db.Column(db.DateTime, nullable=False)
    max_marks = db.Column(db.Float, default=100)
    faculty_id = db.Column(db.Integer, db.ForeignKey('faculty.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AssignmentSubmission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    assignment_id = db.Column(db.Integer, db.ForeignKey('assignment.id'), nullable=False)
    student_id = db.Column(db.String(20), db.ForeignKey('student.student_id'), nullable=False)
    submitted_at = db.Column(db.DateTime)
    marks_obtained = db.Column(db.Float)
    status = db.Column(db.String(20), default='pending')  # pending, submitted, graded, missing
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Exam(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    subject = db.Column(db.String(50), nullable=False)
    class_name = db.Column(db.String(20), nullable=False)
    exam_date = db.Column(db.DateTime, nullable=False)
    max_marks = db.Column(db.Float, nullable=False)
    exam_type = db.Column(db.String(20), nullable=False)  # internal, final, quiz
    faculty_id = db.Column(db.Integer, db.ForeignKey('faculty.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ExamResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    exam_id = db.Column(db.Integer, db.ForeignKey('exam.id'), nullable=False)
    student_id = db.Column(db.String(20), db.ForeignKey('student.student_id'), nullable=False)
    marks_obtained = db.Column(db.Float, nullable=False)
    percentage = db.Column(db.Float)
    grade = db.Column(db.String(5))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Fee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(20), db.ForeignKey('student.student_id'), nullable=False)
    fee_type = db.Column(db.String(50), nullable=False)  # tuition, library, lab, hostel
    amount = db.Column(db.Float, nullable=False)
    due_date = db.Column(db.Date, nullable=False)
    paid_amount = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(20), default='pending')  # pending, partial, paid, overdue
    payment_date = db.Column(db.Date)
    academic_year = db.Column(db.String(10), nullable=False)
    semester = db.Column(db.String(10))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Timetable(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    class_name = db.Column(db.String(20), nullable=False)
    subject = db.Column(db.String(50), nullable=False)
    faculty_id = db.Column(db.Integer, db.ForeignKey('faculty.id'), nullable=False)
    day_of_week = db.Column(db.String(10), nullable=False)  # monday, tuesday, etc.
    start_time = db.Column(db.Time, nullable=False)
    end_time = db.Column(db.Time, nullable=False)
    room_number = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class RiskAssessment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(20), db.ForeignKey('student.student_id'), nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    risk_score = db.Column(db.Integer, nullable=False)  # Changed to Integer for new scoring system
    attendance_risk = db.Column(db.Boolean, default=False)
    assignment_risk = db.Column(db.Boolean, default=False)
    marks_risk = db.Column(db.Boolean, default=False)
    fee_risk = db.Column(db.Boolean, default=False)
    factors = db.Column(db.Text)  # JSON string of detailed risk factors
    recommendations = db.Column(db.Text)  # Academic and financial recommendations
    assessment_date = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'))

# Risk Scoring Functions
def calculate_attendance_percentage(student_id, days=30):
    """Calculate attendance percentage for the last N days"""
    from_date = datetime.now().date() - timedelta(days=days)
    
    total_records = AttendanceRecord.query.filter(
        AttendanceRecord.student_id == student_id,
        AttendanceRecord.date >= from_date
    ).count()
    
    if total_records == 0:
        return 100  # No records means no risk
    
    present_records = AttendanceRecord.query.filter(
        AttendanceRecord.student_id == student_id,
        AttendanceRecord.date >= from_date,
        AttendanceRecord.status == 'present'
    ).count()
    
    return (present_records / total_records) * 100

def count_missing_assignments(student_id):
    """Count missing assignments for a student"""
    # Count explicitly marked missing assignments
    missing_count = AssignmentSubmission.query.filter(
        AssignmentSubmission.student_id == student_id,
        AssignmentSubmission.status == 'missing'
    ).count()
    
    # Count assignments that are past due and not submitted by this specific student
    current_time = datetime.now()
    
    # Get all assignments that are past due
    past_due_assignment_ids = db.session.query(Assignment.id).filter(
        Assignment.due_date < current_time
    ).subquery()
    
    # Get submissions by this student for past due assignments
    student_submissions = db.session.query(AssignmentSubmission.assignment_id).filter(
        AssignmentSubmission.student_id == student_id,
        AssignmentSubmission.assignment_id.in_(past_due_assignment_ids)
    ).subquery()
    
    # Count past due assignments with no submission from this student
    unsubmitted_count = db.session.query(past_due_assignment_ids.c.id).filter(
        past_due_assignment_ids.c.id.notin_(student_submissions)
    ).count()
    
    return missing_count + unsubmitted_count

def calculate_internal_marks_percentage(student_id):
    """Calculate average internal marks percentage"""
    internal_exams = db.session.query(ExamResult).join(Exam).filter(
        ExamResult.student_id == student_id,
        Exam.exam_type == 'internal'
    ).all()
    
    if not internal_exams:
        return 100  # No records means no risk
    
    total_percentage = sum(result.percentage or 0 for result in internal_exams)
    return total_percentage / len(internal_exams)

def check_fee_status(student_id):
    """Check if student has pending or overdue fees"""
    current_date = datetime.now().date()
    
    # Check for overdue fees
    overdue_fees = Fee.query.filter(
        Fee.student_id == student_id,
        Fee.due_date < current_date,
        Fee.status.in_(['pending', 'partial'])
    ).count()
    
    # Check for pending fees
    pending_fees = Fee.query.filter(
        Fee.student_id == student_id,
        Fee.status == 'pending'
    ).count()
    
    return overdue_fees > 0 or pending_fees > 0

def update_student_risk_score(student_id):
    """Update risk score for a student using ML model"""
    student = Student.query.filter_by(student_id=student_id).first()
    if not student:
        return
    
    # Use ML model for prediction
    predicted_level, confidence, explanation = ml_model.predict_risk(student_id)
    
    # Convert ML prediction to numeric score for compatibility
    risk_level_to_score = {'safe': 0, 'warning': 2, 'high_risk': 4}
    ml_risk_score = risk_level_to_score.get(predicted_level, 2)
    
    # Extract individual risk factors for detailed assessment
    attendance_percentage = calculate_attendance_percentage(student_id)
    missing_assignments = count_missing_assignments(student_id)
    internal_marks = calculate_internal_marks_percentage(student_id)
    fee_risk = check_fee_status(student_id)
    
    attendance_risk = attendance_percentage < 70
    assignment_risk = missing_assignments > 1
    marks_risk = internal_marks < 50
    
    # Update student record with ML predictions
    student.risk_score = ml_risk_score
    student.current_risk_level = predicted_level
    
    # Create or update risk assessment with ML insights
    assessment = RiskAssessment.query.filter_by(
        student_id=student_id
    ).order_by(RiskAssessment.assessment_date.desc()).first()
    
    # Only create new assessment if risk level changed or it's been more than a day
    create_new = (not assessment or 
                 assessment.risk_level != predicted_level or
                 (datetime.now() - assessment.assessment_date).days >= 1)
    
    if create_new:
        # Generate recommendations based on ML explanation
        ml_factors = [f"ML Prediction: {explanation}", f"Confidence: {confidence:.2f}"]
        
        new_assessment = RiskAssessment(
            student_id=student_id,
            risk_level=predicted_level,
            risk_score=ml_risk_score,
            attendance_risk=attendance_risk,
            assignment_risk=assignment_risk,
            marks_risk=marks_risk,
            fee_risk=fee_risk,
            factors=json.dumps(ml_factors),
            recommendations=f"ML-driven insights: {explanation}. " + generate_ml_recommendations(predicted_level, confidence)
        )
        db.session.add(new_assessment)
    
    db.session.commit()
    return ml_risk_score

def generate_ml_recommendations(risk_level, confidence):
    """Generate ML-driven recommendations"""
    if risk_level == 'safe':
        return f"Continue monitoring. Model confidence: {confidence:.2f}. Maintain current support level."
    elif risk_level == 'warning':
        return f"Moderate intervention needed. Model confidence: {confidence:.2f}. Schedule check-in and provide targeted support."
    else:
        return f"Urgent intervention required. Model confidence: {confidence:.2f}. Immediate academic and financial counseling recommended."

# Machine Learning Model for Student Risk Prediction

class StudentRiskMLModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = ['attendance_rate', 'missing_assignments', 'avg_marks_percentage', 
                             'fee_status_numeric', 'days_since_enrollment', 'subjects_count']
        self.model_path = 'instance/student_risk_model.pkl'
        self.scaler_path = 'instance/scaler.pkl'
        
    def extract_features_for_student(self, student_id):
        """Extract ML features for a student from database"""
        student = Student.query.filter_by(student_id=student_id).first()
        if not student:
            return None
            
        # Feature 1: Attendance rate (last 30 days)
        attendance_rate = calculate_attendance_percentage(student_id, days=30)
        
        # Feature 2: Missing assignments count
        missing_assignments = count_missing_assignments(student_id)
        
        # Feature 3: Average internal marks percentage
        avg_marks = calculate_internal_marks_percentage(student_id)
        
        # Feature 4: Fee status (numeric: 0=paid, 1=pending, 2=overdue)
        fee_status_numeric = 0
        if check_fee_status(student_id):
            overdue_fees = Fee.query.filter(
                Fee.student_id == student_id,
                Fee.due_date < datetime.now().date(),
                Fee.status.in_(['pending', 'partial'])
            ).count()
            fee_status_numeric = 2 if overdue_fees > 0 else 1
        
        # Feature 5: Days since enrollment
        days_since_enrollment = (datetime.now() - student.created_at).days
        
        # Feature 6: Number of subjects (from test scores)
        subjects_count = db.session.query(TestScore.subject).filter_by(
            student_id=student_id
        ).distinct().count()
        
        features = np.array([
            attendance_rate, missing_assignments, avg_marks, 
            fee_status_numeric, days_since_enrollment, subjects_count
        ]).reshape(1, -1)
        
        return features
    
    def prepare_training_data(self):
        """Prepare training data from database with realistic labels"""
        students = Student.query.all()
        if len(students) < 10:
            # Generate synthetic training data for demo
            return self._generate_synthetic_data()
            
        X = []
        y = []
        
        for student in students:
            features = self.extract_features_for_student(student.student_id)
            if features is not None:
                X.append(features.flatten())
                
                # Create realistic labels based on feature combinations
                attendance_rate, missing_assignments, avg_marks, fee_status, days_enrolled, subjects = features.flatten()
                
                # ML-based risk scoring (more sophisticated than rules)
                risk_score = 0
                if attendance_rate < 70: risk_score += 2
                if missing_assignments > 2: risk_score += 2  
                if avg_marks < 50: risk_score += 2
                if fee_status >= 1: risk_score += 1
                if subjects < 3: risk_score += 1  # Limited subject engagement
                
                # Convert to risk level
                if risk_score <= 2:
                    label = 0  # Safe
                elif risk_score <= 4:
                    label = 1  # Warning 
                else:
                    label = 2  # High Risk
                    
                y.append(label)
        
        return np.array(X), np.array(y)
    
    def _generate_synthetic_data(self, n_samples=200):
        """Generate synthetic training data for demonstration"""
        np.random.seed(42)
        
        # Generate realistic feature distributions
        attendance_rates = np.random.beta(8, 2, n_samples) * 100  # Skewed towards higher attendance
        missing_assignments = np.random.poisson(1.5, n_samples)  # Low count with occasional high
        avg_marks = np.random.normal(75, 15, n_samples)  # Normal distribution around 75%
        avg_marks = np.clip(avg_marks, 0, 100)
        fee_status = np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.2, 0.1])  # Most students paid
        days_enrolled = np.random.normal(180, 60, n_samples)  # ~6 months average
        days_enrolled = np.clip(days_enrolled, 30, 365)
        subjects_count = np.random.choice([3, 4, 5, 6], n_samples, p=[0.1, 0.3, 0.4, 0.2])
        
        X = np.column_stack([
            attendance_rates, missing_assignments, avg_marks, 
            fee_status, days_enrolled, subjects_count
        ])
        
        # Generate labels with realistic correlations
        y = []
        for i in range(n_samples):
            risk_score = 0
            if X[i][0] < 70: risk_score += 2  # Low attendance
            if X[i][1] > 2: risk_score += 2   # Many missing assignments
            if X[i][2] < 50: risk_score += 2  # Low marks
            if X[i][3] >= 1: risk_score += 1  # Fee issues
            if X[i][5] < 4: risk_score += 1   # Few subjects
            
            # Add some randomness to make it more realistic
            risk_score += np.random.normal(0, 0.5)
            
            if risk_score <= 2:
                y.append(0)  # Safe
            elif risk_score <= 4:
                y.append(1)  # Warning
            else:
                y.append(2)  # High Risk
                
        return X, np.array(y)
    
    def train_model(self, save_model=True):
        """Train the ML model on student data"""
        print("Training student risk prediction model...")
        
        X, y = self.prepare_training_data()
        
        if len(X) == 0:
            print("No training data available")
            return False
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained successfully! Accuracy: {accuracy:.3f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Safe', 'Warning', 'High Risk']))
        
        self.is_trained = True
        
        if save_model:
            self.save_model()
            
        return True
    
    def predict_risk(self, student_id):
        """Predict risk level for a student using ML model"""
        if not self.is_trained:
            if not self.load_model():
                # Fallback to rule-based system
                return self._rule_based_prediction(student_id)
                
        features = self.extract_features_for_student(student_id)
        if features is None:
            return 'unknown', 0.0, 'Insufficient data'
            
        # Scale features and predict
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        confidence = np.max(self.model.predict_proba(features_scaled))
        
        risk_levels = ['safe', 'warning', 'high_risk']
        predicted_level = risk_levels[prediction]
        
        # Get feature importance explanation
        feature_importance = self.model.feature_importances_
        top_factors = []
        feature_values = features.flatten()
        
        for i, (name, value, importance) in enumerate(zip(self.feature_names, feature_values, feature_importance)):
            if importance > 0.1:  # Only show important factors
                top_factors.append(f"{name}: {value:.1f} (weight: {importance:.2f})")
        
        explanation = "Key factors: " + "; ".join(top_factors[:3])
        
        return predicted_level, confidence, explanation
    
    def _rule_based_prediction(self, student_id):
        """Fallback rule-based prediction when ML model unavailable"""
        attendance_rate = calculate_attendance_percentage(student_id)
        missing_assignments = count_missing_assignments(student_id)
        avg_marks = calculate_internal_marks_percentage(student_id)
        fee_issues = check_fee_status(student_id)
        
        risk_score = 0
        factors = []
        
        if attendance_rate < 70:
            risk_score += 2
            factors.append(f"Low attendance: {attendance_rate:.1f}%")
        if missing_assignments > 1:
            risk_score += 2
            factors.append(f"Missing assignments: {missing_assignments}")
        if avg_marks < 50:
            risk_score += 2
            factors.append(f"Low marks: {avg_marks:.1f}%")
        if fee_issues:
            risk_score += 1
            factors.append("Fee issues")
        
        if risk_score <= 2:
            level = 'safe'
        elif risk_score <= 4:
            level = 'warning'
        else:
            level = 'high_risk'
            
        explanation = "Rule-based: " + "; ".join(factors) if factors else "No major risk factors"
        confidence = min(0.8, (risk_score / 6))  # Approximate confidence
        
        return level, confidence, explanation
    
    def save_model(self):
        """Save trained model to disk"""
        try:
            os.makedirs('instance', exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self):
        """Load trained model from disk"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.is_trained = True
                print("Model loaded successfully")
                return True
        except Exception as e:
            print(f"Error loading model: {e}")
        return False

# Global ML model instance
ml_model = StudentRiskMLModel()

def generate_recommendations(risk_factors):
    """Generate academic and financial recommendations based on risk factors"""
    recommendations = []
    
    for factor in risk_factors:
        if "attendance" in factor.lower():
            recommendations.append("Academic: Schedule regular check-ins, provide attendance incentives")
        elif "assignment" in factor.lower():
            recommendations.append("Academic: Provide assignment reminders, offer extended deadlines if needed")
        elif "marks" in factor.lower():
            recommendations.append("Academic: Arrange tutoring sessions, review study methods")
        elif "fee" in factor.lower():
            recommendations.append("Financial: Contact for payment plan, discuss scholarship opportunities")
    
    if not recommendations:
        recommendations.append("Continue current support and monitor progress")
    
    return "; ".join(recommendations)

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.role == 'admin':
            return redirect(url_for('admin_dashboard'))
        elif current_user.role == 'faculty':
            return redirect(url_for('faculty_dashboard'))
        else:
            return redirect(url_for('student_dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if username looks like a student ID (e.g., 24uam101)
        if re.match(r'^\d{2}[a-zA-Z]{3,4}\d{3}$', username.lower()):
            # Student login with student_id
            student = Student.query.filter_by(student_id=username.lower()).first()
            if student:
                # Create or get user account for student
                user = User.query.filter_by(username=username.lower()).first()
                if not user:
                    user = User(
                        username=username.lower(),
                        email=student.email or f"{username.lower()}@student.edu",
                        role='student'
                    )
                    user.set_password(password)  # Set the password they provided
                    db.session.add(user)
                    db.session.commit()
                elif user.check_password(password):
                    login_user(user)
                    flash('Student login successful!', 'success')
                    return redirect(url_for('student_dashboard'))
                else:
                    flash('Invalid password for student ID!', 'error')
            else:
                flash('Student ID not found! Please contact administration.', 'error')
        else:
            # Regular user login
            user = User.query.filter_by(username=username).first()
            
            if user and user.check_password(password):
                login_user(user)
                flash('Login successful!', 'success')
                return redirect(url_for('index'))
            else:
                flash('Invalid username or password!', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        role = request.form.get('role', 'student')  # Allow faculty registration
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'error')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered!', 'error')
            return render_template('register.html')
        
        # Create new user
        user = User(username=username, email=email, role=role)
        user.set_password(password)
        
        db.session.add(user)
        db.session.flush()  # Get the user ID
        
        # Create Faculty record if registering as faculty
        if role == 'faculty':
            faculty = Faculty(
                faculty_id=f"FAC{user.id:04d}",
                name=username,  # Use username as name initially
                email=email,
                user_id=user.id
            )
            db.session.add(faculty)
        
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        flash('Access denied!', 'error')
        return redirect(url_for('index'))
    
    # Get student statistics
    total_students = Student.query.count()
    safe_count = Student.query.filter_by(current_risk_level='safe').count()
    warning_count = Student.query.filter_by(current_risk_level='warning').count()
    high_risk_count = Student.query.filter_by(current_risk_level='high_risk').count()
    
    # Get recent risk assessments
    recent_assessments = RiskAssessment.query.order_by(RiskAssessment.assessment_date.desc()).limit(10).all()
    
    # Get all students with their risk levels
    students = Student.query.all()
    
    return render_template('admin_dashboard.html', 
                         total_students=total_students,
                         safe_count=safe_count,
                         warning_count=warning_count,
                         high_risk_count=high_risk_count,
                         students=students,
                         recent_assessments=recent_assessments)

@app.route('/faculty/dashboard')
@login_required
def faculty_dashboard():
    if current_user.role != 'faculty':
        flash('Access denied!', 'error')
        return redirect(url_for('index'))
    
    # Get all students and their risk assessments
    students = Student.query.all()
    
    # Update risk scores for all students
    for student in students:
        update_student_risk_score(student.student_id)
    
    # Get updated statistics
    total_students = len(students)
    safe_count = Student.query.filter_by(current_risk_level='safe').count()
    warning_count = Student.query.filter_by(current_risk_level='warning').count()
    high_risk_count = Student.query.filter_by(current_risk_level='high_risk').count()
    
    # Get recent risk assessments
    recent_assessments = RiskAssessment.query.order_by(RiskAssessment.assessment_date.desc()).limit(10).all()
    
    # --- REMOVE assignments query and assignments=assignments ---
    return render_template('faculty_dashboard.html', 
                         students=students,
                         total_students=total_students,
                         safe_count=safe_count,
                         warning_count=warning_count,
                         high_risk_count=high_risk_count,
                         recent_assessments=recent_assessments)

@app.route('/student/dashboard')
@login_required
def student_dashboard():
    # Allow both student role and student_id login
    student = None
    if current_user.role == 'student':
        # Try to find student by username (student_id)
        student = Student.query.filter_by(student_id=current_user.username).first()
    elif hasattr(current_user, 'student_id'):
        student = Student.query.filter_by(student_id=current_user.student_id).first()
    
    if not student:
        # Create a basic student record if not found
        student = Student(
            student_id=current_user.username,
            name=current_user.username.upper(),
            class_name='AIML',
            email=current_user.email
        )
        db.session.add(student)
        db.session.commit()
    
    # Get student's academic data
    attendance_percentage = calculate_attendance_percentage(student.student_id)
    missing_assignments = count_missing_assignments(student.student_id)
    internal_marks = calculate_internal_marks_percentage(student.student_id)
    
    # Get recent activities
    recent_tests = TestScore.query.filter_by(student_id=student.student_id)\
                                 .order_by(TestScore.test_date.desc()).limit(5).all()
    recent_attendance = AttendanceRecord.query.filter_by(student_id=student.student_id)\
                                             .order_by(AttendanceRecord.date.desc()).limit(5).all()
    
    return render_template('student_dashboard.html', 
                         student=student,
                         attendance_percentage=attendance_percentage,
                         missing_assignments=missing_assignments,
                         internal_marks=internal_marks,
                         recent_tests=recent_tests,
                         recent_attendance=recent_attendance)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_data():
    if current_user.role.lower() not in ['admin', 'faculty']:
        flash('Access denied!', 'error')
        return redirect(url_for('index'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected!', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        data_type = request.form.get('data_type')
        action = request.form.get('action')  # 'replace' or 'update'
        
        if not file.filename:
            flash('No file selected!', 'error')
            return redirect(request.url)
        
        if not action:
            flash('Please select an action (replace or update)!', 'error')
            return redirect(request.url)
        
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                if data_type == 'students':
                    process_student_data(filepath, action)
                elif data_type == 'attendance':
                    process_attendance_data(filepath, action)
                elif data_type == 'test_scores':
                    process_test_scores_data(filepath, action)
                
                flash(f'File uploaded and {action}d successfully!', 'success')
                os.remove(filepath)  # Clean up
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
                if os.path.exists(filepath):
                    os.remove(filepath)
        else:
            flash('Invalid file format. Please upload CSV or Excel files.', 'error')
    
    return render_template('upload.html')


# Faculty Management Routes
@app.route('/faculty/timetable')
@login_required
def manage_timetable():
    if current_user.role != 'faculty':
        flash('Access denied!', 'error')
        return redirect(url_for('index'))
    
    faculty = Faculty.query.filter_by(user_id=current_user.id).first()
    if not faculty:
        flash('Faculty profile not found!', 'error')
        return redirect(url_for('index'))
    
    timetable = Timetable.query.filter_by(faculty_id=faculty.id).all()
    return render_template('manage_timetable.html', timetable=timetable)

@app.route('/faculty/assignments')
@login_required
def manage_assignments():
    if current_user.role != 'faculty':
        flash('Access denied!', 'error')
        return redirect(url_for('index'))
    
    faculty = Faculty.query.filter_by(user_id=current_user.id).first()
    if not faculty:
        flash('Faculty profile not found!', 'error')
        return redirect(url_for('index'))
    
    assignments = Assignment.query.filter_by(faculty_id=faculty.id)\
                                  .order_by(Assignment.created_at.desc()).all()
    return render_template('manage_assignments.html', assignments=assignments)

@app.route('/faculty/exams')
@login_required  
def manage_exams():
    if current_user.role != 'faculty':
        flash('Access denied!', 'error')
        return redirect(url_for('index'))
    
    faculty = Faculty.query.filter_by(user_id=current_user.id).first()
    if not faculty:
        flash('Faculty profile not found!', 'error')
        return redirect(url_for('index'))
        
    exams = Exam.query.filter_by(faculty_id=faculty.id).all()
    return render_template('manage_exams.html', exams=exams)


@app.route('/faculty/fees')
@login_required
def manage_fees():
    if current_user.role != 'faculty':
        flash('Access denied!', 'error')
        return redirect(url_for('index'))
        
    fees = Fee.query.all()  # Faculty can view all fees for management
    return render_template('manage_fees.html', fees=fees)

@app.route('/faculty/assignments/create', methods=['GET', 'POST'])
@login_required
def create_assignment():
    if current_user.role != 'faculty':
        flash('Access denied!', 'error')
        return redirect(url_for('index'))
    
    faculty = Faculty.query.filter_by(user_id=current_user.id).first()
    if not faculty:
        flash('Faculty profile not found!', 'error')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        # Optional: handle file upload if needed
        file = request.files.get('file')
        filename = None
        if file:
            filename = f"uploads/{file.filename}"
            file.save(f"static/{filename}")
        
        assignment = Assignment(
            title=request.form['title'],
            description=request.form.get('description', ''),
            subject=request.form['subject'],
            class_name=request.form['class_name'],
            due_date=datetime.strptime(request.form['due_date'], '%Y-%m-%dT%H:%M'),
            max_marks=float(request.form.get('max_marks', 100)),
            faculty_id=faculty.id,
            file_path=filename
        )
        db.session.add(assignment)
        db.session.commit()
        flash('Assignment created successfully!', 'success')
        return redirect(url_for('manage_assignments'))
    
    return render_template('create_assignment.html')

# Enhanced CRUD Operations for Faculty Dashboard

@app.route('/faculty/exams/create', methods=['GET', 'POST'])
@login_required
def create_exam():
    if current_user.role != 'faculty':
        flash('Access denied!', 'error')
        return redirect(url_for('index'))
    
    faculty = Faculty.query.filter_by(user_id=current_user.id).first()
    if not faculty:
        flash('Faculty profile not found!', 'error')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        exam = Exam(
            name=request.form['name'],
            subject=request.form['subject'],
            class_name=request.form['class_name'],
            exam_date=datetime.strptime(request.form['exam_date'], '%Y-%m-%dT%H:%M'),
            max_marks=float(request.form['max_marks']),
            exam_type=request.form['exam_type'],
            faculty_id=faculty.id
        )
        db.session.add(exam)
        db.session.commit()
        flash('Exam created successfully!', 'success')
        return redirect(url_for('manage_exams'))
    
    return render_template('create_exam.html')

@app.route('/faculty/exams/edit/<int:exam_id>', methods=['GET', 'POST'])
@login_required
def edit_exam(exam_id):
    if current_user.role != 'faculty':
        flash('Access denied!', 'error')
        return redirect(url_for('index'))
    
    faculty = Faculty.query.filter_by(user_id=current_user.id).first()
    if not faculty:
        flash('Faculty profile not found!', 'error')
        return redirect(url_for('index'))
    
    exam = Exam.query.filter_by(id=exam_id, faculty_id=faculty.id).first()
    if not exam:
        flash('Exam not found!', 'error')
        return redirect(url_for('manage_exams'))
    
    if request.method == 'POST':
        exam.name = request.form['name']
        exam.subject = request.form['subject']
        exam.class_name = request.form['class_name']
        exam.exam_date = datetime.strptime(request.form['exam_date'], '%Y-%m-%dT%H:%M')
        exam.max_marks = float(request.form['max_marks'])
        exam.exam_type = request.form['exam_type']
        db.session.commit()
        flash('Exam updated successfully!', 'success')
        return redirect(url_for('manage_exams'))
    
    return render_template('edit_exam.html', exam=exam)



@app.route('/faculty/exams/delete/<int:exam_id>', methods=['POST'])
@login_required
def delete_exam(exam_id):
    if current_user.role != 'faculty':
        flash('Access denied!', 'error')
        return redirect(url_for('index'))
    
    faculty = Faculty.query.filter_by(user_id=current_user.id).first()
    if not faculty:
        flash('Faculty profile not found!', 'error')
        return redirect(url_for('index'))
    
    exam = Exam.query.filter_by(id=exam_id, faculty_id=faculty.id).first()
    if exam:
        # Delete related exam results first
        ExamResult.query.filter_by(exam_id=exam.id).delete()
        db.session.delete(exam)
        db.session.commit()
        flash('Exam deleted successfully!', 'success')
    else:
        flash('Exam not found!', 'error')
    
    return redirect(url_for('manage_exams'))
# AI Chatbot Routes
@app.route('/api/chat', methods=['POST'])
@login_required
def chat_api():
    """AI Chatbot API endpoint"""
    data = request.get_json()
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Get student data for personalized responses
    student = None
    if current_user.role == 'student':
        student = Student.query.filter_by(student_id=current_user.username).first()
    
    # Generate AI response
    bot_response = generate_ai_response(user_message, student)
    
    return jsonify({
        'response': bot_response,
        'timestamp': datetime.now().isoformat()
    })

def generate_ai_response(message, student=None):
    """Generate AI counselor response based on user message and student data"""
    message_lower = message.lower()
    
    # Personalized responses based on student data
    if student:
        attendance = calculate_attendance_percentage(student.student_id)
        missing_assignments = count_missing_assignments(student.student_id)
        internal_marks = calculate_internal_marks_percentage(student.student_id)
        
        # Attendance-related queries
        if any(word in message_lower for word in ['attendance', 'absent', 'present', 'class']):
            if attendance < 70:
                return f"I see your attendance is at {attendance:.1f}%, which is below the recommended 75%. Here are some tips to improve: 1) Set daily reminders for classes, 2) Find a study buddy for accountability, 3) Talk to your professors about any challenges you're facing. Would you like specific strategies for better time management?"
            else:
                return f"Great job! Your attendance is at {attendance:.1f}%. Keep maintaining this consistency. Regular attendance is key to academic success."
        
        # Assignment-related queries
        if any(word in message_lower for word in ['assignment', 'homework', 'submit', 'deadline']):
            if missing_assignments > 0:
                return f"I notice you have {missing_assignments} pending assignments. Let's create a plan: 1) List all pending work with deadlines, 2) Break large tasks into smaller chunks, 3) Set daily goals, 4) Reward yourself for completing tasks. Need help prioritizing your assignments?"
            else:
                return "Excellent! You're up to date with your assignments. This shows great time management skills. Keep this momentum going!"
        
        # Grades/marks related queries
        if any(word in message_lower for word in ['marks', 'grades', 'score', 'test', 'exam']):
            if internal_marks < 60:
                return f"Your current average is {internal_marks:.1f}%. Let's work on improvement strategies: 1) Review your study methods, 2) Form study groups, 3) Seek help from professors during office hours, 4) Practice active recall techniques. What subject would you like to focus on first?"
            else:
                return f"You're doing well with an average of {internal_marks:.1f}%! To maintain or improve further: 1) Continue your current study routine, 2) Challenge yourself with advanced problems, 3) Help classmates to reinforce your learning."
    
    # General motivational and study-related responses
    responses = {
        'stress': [
            "It's normal to feel stressed sometimes. Try these techniques: 1) Deep breathing exercises, 2) Take short breaks every hour, 3) Talk to friends or counselors, 4) Maintain a regular sleep schedule. Remember, you're not alone in this journey!",
            "Stress can be overwhelming, but you can manage it! Consider: 1) Breaking tasks into smaller parts, 2) Practicing mindfulness, 3) Regular exercise, 4) Proper time management. Would you like specific stress-relief techniques?"
        ],
        'motivation': [
            "Remember why you started this journey! Every small step counts. Set small, achievable goals and celebrate your progress. You have the potential to succeed!",
            "Motivation comes and goes, but discipline stays. Create a routine, stick to it, and trust the process. Your future self will thank you for the effort you put in today!"
        ],
        'study': [
            "Effective studying is about quality, not just quantity. Try: 1) Active recall instead of just re-reading, 2) Spaced repetition, 3) Teaching concepts to others, 4) Taking regular breaks. What subject are you focusing on?",
            "Great study habits include: 1) Creating a distraction-free environment, 2) Using the Pomodoro technique, 3) Making summary notes, 4) Regular self-testing. Need help with any specific topic?"
        ],
        'career': [
            "Your career path is unique to you! Consider: 1) Your interests and strengths, 2) Industry trends, 3) Networking opportunities, 4) Continuous learning. What field interests you most?",
            "Career planning is exciting! Start by: 1) Exploring different options, 2) Talking to professionals in your field of interest, 3) Building relevant skills, 4) Gaining practical experience through internships."
        ],
        'help': [
            "I'm here to help! I can assist with: 1) Study strategies, 2) Time management, 3) Stress management, 4) Academic planning, 5) Motivation and goal setting. What would you like to discuss?",
            "You can ask me about: 1) Improving attendance, 2) Assignment management, 3) Study techniques, 4) Career guidance, 5) Dealing with academic stress. How can I support you today?"
        ]
    }
    
    # Match user message to response categories
    for category, category_responses in responses.items():
        if category in message_lower:
            return random.choice(category_responses)
    
    # Default responses for unmatched queries
    default_responses = [
        "That's an interesting question! While I focus on academic counseling, I'd suggest discussing this with your professors or academic advisors for more detailed guidance.",
        "I understand your concern. For academic success, focus on consistent attendance, timely assignment submission, and active participation in class. Is there a specific academic challenge you'd like help with?",
        "Every student's journey is unique. What matters most is your commitment to learning and growth. How can I help you with your academic goals today?",
        "I'm here to support your academic journey! Whether it's study strategies, time management, or motivation, I'm ready to help. What's on your mind?"
    ]
    
    return random.choice(default_responses)

# Quick Actions Routes
@app.route('/faculty/manage_timetable')
@login_required
def manage_timetable():
    if current_user.role != 'faculty':
        flash('Access denied!', 'error')
        return redirect(url_for('index'))
    
    faculty = Faculty.query.filter_by(user_id=current_user.id).first()
    if not faculty:
        flash('Faculty profile not found!', 'error')
        return redirect(url_for('index'))
    
    timetable = Timetable.query.filter_by(faculty_id=faculty.id).all()
    return render_template('manage_timetable.html', timetable=timetable, faculty=faculty)

@app.route('/faculty/send_email')
@login_required
def send_email():
    if current_user.role != 'faculty':
        flash('Access denied!', 'error')
        return redirect(url_for('index'))
    
    # Get high-risk students for email alerts
    high_risk_students = Student.query.filter_by(current_risk_level='high_risk').all()
    warning_students = Student.query.filter_by(current_risk_level='warning').all()
    
    return render_template('send_email.html', 
                         high_risk_students=high_risk_students,
                         warning_students=warning_students)

@app.route('/faculty/generate_report')
@login_required
def generate_report():
    if current_user.role != 'faculty':
        flash('Access denied!', 'error')
        return redirect(url_for('index'))
    
    # Generate comprehensive report data
    total_students = Student.query.count()
    safe_count = Student.query.filter_by(current_risk_level='safe').count()
    warning_count = Student.query.filter_by(current_risk_level='warning').count()
    high_risk_count = Student.query.filter_by(current_risk_level='high_risk').count()
    
    # Get detailed student data
    students = Student.query.all()
    
    return render_template('generate_report.html',
                         total_students=total_students,
                         safe_count=safe_count,
                         warning_count=warning_count,
                         high_risk_count=high_risk_count,
                         students=students)

# Student Profile API for Faculty
@app.route('/api/student/<student_id>')
@login_required
def get_student_profile(student_id):
    if current_user.role not in ['faculty', 'admin']:
        return jsonify({'error': 'Access denied'}), 403
    
    student = Student.query.filter_by(student_id=student_id).first()
    if not student:
        return jsonify({'error': 'Student not found'}), 404
    
    # Get comprehensive student data
    attendance_percentage = calculate_attendance_percentage(student_id)
    missing_assignments = count_missing_assignments(student_id)
    internal_marks = calculate_internal_marks_percentage(student_id)
    fee_status = check_fee_status(student_id)
    
    # Get recent activities
    recent_tests = TestScore.query.filter_by(student_id=student_id)\
                                 .order_by(TestScore.test_date.desc()).limit(5).all()
    recent_attendance = AttendanceRecord.query.filter_by(student_id=student_id)\
                                             .order_by(AttendanceRecord.date.desc()).limit(10).all()
    
    return jsonify({
        'student': {
            'student_id': student.student_id,
            'name': student.name,
            'class_name': student.class_name,
            'email': student.email,
            'phone': student.phone,
            'risk_level': student.current_risk_level,
            'risk_score': student.risk_score
        },
        'metrics': {
            'attendance_percentage': attendance_percentage,
            'missing_assignments': missing_assignments,
            'internal_marks': internal_marks,
            'fee_status': fee_status
        },
        'recent_tests': [{'subject': t.subject, 'score': t.score, 'max_score': t.max_score, 'date': t.test_date.isoformat()} for t in recent_tests],
        'recent_attendance': [{'date': a.date.isoformat(), 'status': a.status, 'subject': a.subject} for a in recent_attendance]
    })

def allowed_file(filename):
    """Check if uploaded file has an allowed extension"""
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_student_data(filepath, action='update'):
    """Process uploaded student data file with replace/update option"""
    import pandas as pd

    # Use app context so SQLAlchemy knows which app to use
    with app.app_context():
        # Load CSV or Excel
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        if action == 'replace':
            # Delete all existing student records
            Student.query.delete()
            db.session.commit()

        for _, row in df.iterrows():
            student_id = str(row['student_id'])
            student = Student.query.filter_by(student_id=student_id).first()

            if student:
                if action == 'update':
                    # Update existing student fields
                    student.name = row['name']
                    student.class_name = row['class']
                    student.email = row.get('email', student.email)
                    student.phone = row.get('phone', student.phone)
                    student.guardian_email = row.get('guardian_email', student.guardian_email)
                    student.guardian_phone = row.get('guardian_phone', student.guardian_phone)
            else:
                # Add new student
                new_student = Student(
                    student_id=student_id,
                    name=row['name'],
                    class_name=row['class'],
                    email=row.get('email', ''),
                    phone=row.get('phone', ''),
                    guardian_email=row.get('guardian_email', ''),
                    guardian_phone=row.get('guardian_phone', '')
                )
                db.session.add(new_student)

        db.session.commit()



def process_attendance_data(filepath,action='update'):
    """Process uploaded attendance data file"""
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    
    for _, row in df.iterrows():
        # Check if record already exists
        existing = AttendanceRecord.query.filter_by(
            student_id=str(row['student_id']),
            date=pd.to_datetime(row['date']).date(),
            subject=row.get('subject', '')
        ).first()
        
        if not existing:
            record = AttendanceRecord(
                student_id=str(row['student_id']),
                date=pd.to_datetime(row['date']).date(),
                status=row['status'],
                subject=row.get('subject', '')
            )
            db.session.add(record)
    
    db.session.commit()
@app.route('/faculty/assignments/edit/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_assignment(id):
    if current_user.role != 'faculty':
        flash('Access denied!', 'error')
        return redirect(url_for('index'))
    
    faculty = Faculty.query.filter_by(user_id=current_user.id).first()
    if not faculty:
        flash('Faculty profile not found!', 'error')
        return redirect(url_for('index'))
    
    assignment = Assignment.query.filter_by(id=id, faculty_id=faculty.id).first()
    if not assignment:
        flash('Assignment not found!', 'error')
        return redirect(url_for('manage_assignments'))
    
    if request.method == 'POST':
        assignment.title = request.form['title']
        assignment.description = request.form.get('description', '')
        assignment.subject = request.form['subject']
        assignment.class_name = request.form['class_name']
        assignment.due_date = datetime.strptime(request.form['due_date'], '%Y-%m-%dT%H:%M')
        assignment.max_marks = float(request.form.get('max_marks', 100))
        db.session.commit()
        flash('Assignment updated successfully!', 'success')
        return redirect(url_for('manage_assignments'))
    
    return render_template('edit_assignment.html', assignment=assignment)
@app.route('/faculty/assignments/delete/<int:id>', methods=['POST'])
@login_required
def delete_assignment(id):
    if current_user.role != 'faculty':
        flash('Access denied!', 'error')
        return redirect(url_for('index'))
    
    faculty = Faculty.query.filter_by(user_id=current_user.id).first()
    if not faculty:
        flash('Faculty profile not found!', 'error')
        return redirect(url_for('index'))
    
    assignment = Assignment.query.filter_by(id=id, faculty_id=faculty.id).first()
    if not assignment:
        flash('Assignment not found!', 'error')
        return redirect(url_for('manage_assignments'))
    
    db.session.delete(assignment)
    db.session.commit()
    flash('Assignment deleted successfully!', 'success')
    return redirect(url_for('manage_assignments'))

def process_test_scores_data(filepath):
    """Process uploaded test scores data file"""
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    
    for _, row in df.iterrows():
        score = TestScore(
            student_id=str(row['student_id']),
            subject=row['subject'],
            test_name=row['test_name'],
            score=float(row['score']),
            max_score=float(row['max_score']),
            test_date=pd.to_datetime(row['test_date']).date()
        )
        db.session.add(score)
    
    db.session.commit()

# Database initialization (legacy function, replaced by main initialization)
def create_tables():
    """Create database tables (deprecated - use main block initialization)"""
    db.create_all()

# Initialize application when imported
# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
try:
    os.makedirs(app.instance_path, exist_ok=True)
except OSError:
    pass  # Directory already exists or can't create (not critical)

# Initialize database and create admin user with error handling
with app.app_context():
    try:
        # Create all database tables
        db.create_all()
        print("Database tables created successfully!")
        
        # Create default admin user if not exists
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin_username = os.environ.get('ADMIN_USERNAME', 'admin')
            admin_password = os.environ.get('ADMIN_PASSWORD', 'admin123')
            admin = User(username=admin_username, email='admin@school.edu', role='admin')
            admin.set_password(admin_password)
            db.session.add(admin)
            db.session.commit()
            print(f"Created admin user: {admin_username}")
        
        # Initialize ML model - Train if not already trained
        print("Initializing ML-based risk prediction system...")
        try:
            if not ml_model.load_model():
                print("Training ML model for first time...")
                ml_model.train_model()
                print("ML model training completed!")
            else:
                print("ML model loaded successfully!")
                
            # Update risk scores for existing students using ML
            students = Student.query.all()
            if students:
                print(f"Updating ML-based risk scores for {len(students)} students...")
                for student in students:
                    update_student_risk_score(student.student_id)
                print("ML-based risk score updates completed!")
                
        except Exception as ml_error:
            print(f"ML model initialization failed: {ml_error}")
            print("Application will continue without ML predictions")
            
    except Exception as db_error:
        print(f"Database initialization failed: {db_error}")
        print("Application will continue in limited mode")
        # Initialize ML model even if database fails
        try:
            print("Initializing ML model in standalone mode...")
            ml_model.train_model()
            print("ML model training completed in standalone mode!")
        except Exception as ml_error:
            print(f"ML model initialization also failed: {ml_error}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)