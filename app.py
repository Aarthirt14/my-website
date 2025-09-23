from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', 'dev-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(app.instance_path, 'student_prediction.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize extensions
db = SQLAlchemy(app)
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
    role = db.Column(db.String(20), nullable=False, default='student')  # student, mentor, admin
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

class RiskAssessment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(20), db.ForeignKey('student.student_id'), nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    risk_score = db.Column(db.Float, nullable=False)
    factors = db.Column(db.Text)  # JSON string of risk factors
    assessment_date = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'))

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.role == 'admin':
            return redirect(url_for('admin_dashboard'))
        elif current_user.role == 'mentor':
            return redirect(url_for('mentor_dashboard'))
        else:
            return redirect(url_for('student_dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
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
        role = 'student'  # Force all public registrations to be students
        
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

@app.route('/mentor/dashboard')
@login_required
def mentor_dashboard():
    if current_user.role != 'mentor':
        flash('Access denied!', 'error')
        return redirect(url_for('index'))
    
    # Get students assigned to this mentor (for now, show all high-risk students)
    high_risk_students = Student.query.filter_by(current_risk_level='high_risk').all()
    warning_students = Student.query.filter_by(current_risk_level='warning').all()
    
    return render_template('mentor_dashboard.html', 
                         high_risk_students=high_risk_students,
                         warning_students=warning_students)

@app.route('/student/dashboard')
@login_required
def student_dashboard():
    if current_user.role != 'student':
        flash('Access denied!', 'error')
        return redirect(url_for('index'))
    
    # For now, show a simple student dashboard
    # In a real implementation, you'd link User to Student via student_id
    return render_template('student_dashboard.html')

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_data():
    if current_user.role != 'admin':
        flash('Access denied!', 'error')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected!', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        data_type = request.form['data_type']
        
        if not file.filename:
            flash('No file selected!', 'error')
            return redirect(request.url)
        
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the uploaded file
            try:
                if data_type == 'students':
                    process_student_data(filepath)
                elif data_type == 'attendance':
                    process_attendance_data(filepath)
                elif data_type == 'test_scores':
                    process_test_scores_data(filepath)
                
                flash('File uploaded and processed successfully!', 'success')
                os.remove(filepath)  # Clean up uploaded file
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
                if os.path.exists(filepath):
                    os.remove(filepath)
        else:
            flash('Invalid file format. Please upload CSV or Excel files.', 'error')
    
    return render_template('upload.html')

def allowed_file(filename):
    """Check if uploaded file has an allowed extension"""
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_student_data(filepath):
    """Process uploaded student data file"""
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    
    for _, row in df.iterrows():
        student = Student.query.filter_by(student_id=str(row['student_id'])).first()
        if not student:
            student = Student(
                student_id=str(row['student_id']),
                name=row['name'],
                class_name=row['class'],
                email=row.get('email', ''),
                phone=row.get('phone', ''),
                guardian_email=row.get('guardian_email', ''),
                guardian_phone=row.get('guardian_phone', '')
            )
            db.session.add(student)
    
    db.session.commit()

def process_attendance_data(filepath):
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

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.instance_path, exist_ok=True)
    
    # Initialize database and create admin user
    with app.app_context():
        # Create all database tables
        db.create_all()
        
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
    
    app.run(host='0.0.0.0', port=5000, debug=True)