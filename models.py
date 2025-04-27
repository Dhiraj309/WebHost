from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class User(db.Model, UserMixin):
    __tablename__ = 'Users'  # This tells SQLAlchemy to use the 'Users' table in the database
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)  # Adjust based on your table
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'

    # Flask-Login requires these methods to work
    def get_id(self):
        """Return the user ID as a string"""
        return str(self.id)

    def is_active(self):
        """Return True if the user is active (you can modify logic as needed)"""
        return True  # Assuming all users are active

    def is_authenticated(self):
        """Return True if the user is authenticated (logged in)"""
        return True  # This should always return True for authenticated users

    def is_anonymous(self):
        """Return False since the user is not anonymous"""
        return False  # This should always return False for authenticated users
