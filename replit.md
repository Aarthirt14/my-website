# Student Prediction System - Replit Setup

## Overview
A Flask-based student prediction system for academic risk assessment. The application predicts student performance based on attendance, assignments, test scores, and fees to identify at-risk students early.

## Current State
- Successfully migrated from SQLite to PostgreSQL for better performance
- Running on Flask development server with proper proxy configuration  
- Admin user created with default credentials (admin/admin123)
- All dependencies installed and working correctly

## Recent Changes (2025-09-25)
- Configured Flask app for Replit environment with ProxyFix middleware
- Migrated database from SQLite to PostgreSQL using DATABASE_URL
- Set up proper workflow on port 5000 with webview output
- Added database engine options for connection pooling
- Configured deployment settings for autoscale with gunicorn

## User Preferences
- Default development setup maintained
- PostgreSQL preferred over SQLite for production-like testing
- Frontend workflow priority for immediate user visibility

## Project Architecture
### Backend
- **Framework**: Flask with SQLAlchemy ORM
- **Database**: PostgreSQL (via Replit environment)
- **Authentication**: Flask-Login with role-based access (admin, faculty, student)
- **File Uploads**: Secure file handling for Excel/CSV data

### Frontend
- **Templates**: Jinja2 with Bootstrap styling
- **Static Assets**: CSS and JavaScript in /static directory
- **Responsive Design**: Mobile-friendly interface

### Key Models
- User (authentication)
- Student (core student data)  
- AttendanceRecord, TestScore, Fee (performance metrics)
- RiskAssessment (predictive analysis results)
- Faculty, Assignment, Exam (academic management)

### Risk Assessment Algorithm
- Attendance Risk: <70% attendance
- Assignment Risk: >1 missing assignments  
- Marks Risk: <50% internal marks
- Fee Risk: Pending/overdue payments
- Scoring: 0=safe, 1-2=warning, 3+=high_risk

## Environment Configuration
- **Port**: 5000 (webview output)
- **Database**: PostgreSQL via DATABASE_URL
- **Secrets**: SESSION_SECRET configured
- **Admin Defaults**: admin/admin123 (change in production)

## File Structure
- `app.py` - Main Flask application
- `templates/` - HTML templates with role-based dashboards  
- `static/` - CSS/JS assets
- `instance/` - Legacy SQLite database (now unused)
- `uploads/` - File upload directory (auto-created)

## Deployment
- **Target**: Autoscale deployment
- **Server**: Gunicorn with proper binding
- **Development**: Flask dev server with debug mode