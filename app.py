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
    """Update risk score for a student based on enhanced algorithm"""
    student = Student.query.filter_by(student_id=student_id).first()
    if not student:
        return
    
    risk_score = 0
    risk_factors = []
    
    # 1. Attendance Risk (+1 if <70%)
    attendance_percentage = calculate_attendance_percentage(student_id)
    attendance_risk = attendance_percentage < 70
    if attendance_risk:
        risk_score += 1
        risk_factors.append(f"Low attendance: {attendance_percentage:.1f}%")
    
    # 2. Assignment Risk (+1 if >1 missing)
    missing_assignments = count_missing_assignments(student_id)
    assignment_risk = missing_assignments > 1
    if assignment_risk:
        risk_score += 1
        risk_factors.append(f"Missing assignments: {missing_assignments}")
    
    # 3. Internal Marks Risk (+1 if <50%)
    internal_marks = calculate_internal_marks_percentage(student_id)
    marks_risk = internal_marks < 50
    if marks_risk:
        risk_score += 1
        risk_factors.append(f"Low internal marks: {internal_marks:.1f}%")
    
    # 4. Fee Risk (+2 if pending/overdue)
    fee_risk = check_fee_status(student_id)
    if fee_risk:
        risk_score += 2
        risk_factors.append("Pending/overdue fees")
    
    # Determine risk level
    if risk_score == 0:
        risk_level = 'safe'
    elif risk_score <= 2:
        risk_level = 'warning'
    else:
        risk_level = 'high_risk'
    
    # Update student record
    student.risk_score = risk_score
    student.current_risk_level = risk_level
    
    # Create or update risk assessment
    assessment = RiskAssessment.query.filter_by(
        student_id=student_id
    ).order_by(RiskAssessment.assessment_date.desc()).first()
    
    # Only create new assessment if risk level changed or it's been more than a day
    create_new = (not assessment or 
                 assessment.risk_level != risk_level or
                 (datetime.now() - assessment.assessment_date).days >= 1)
    
    if create_new:
        new_assessment = RiskAssessment(
            student_id=student_id,
            risk_level=risk_level,
            risk_score=risk_score,
            attendance_risk=attendance_risk,
            assignment_risk=assignment_risk,
            marks_risk=marks_risk,
            fee_risk=fee_risk,
            factors=json.dumps(risk_factors),
            recommendations=generate_recommendations(risk_factors)
        )
        db.session.add(new_assessment)
    
    db.session.commit()
    return risk_score

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
    
    assignments = Assignment.query.filter_by(faculty_id=faculty.id).all()
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

# Initialize application when imported
# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
try:
    os.makedirs(app.instance_path, exist_ok=True)
except OSError:
    pass  # Directory already exists or can't create (not critical)

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)