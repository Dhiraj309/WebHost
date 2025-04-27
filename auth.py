# from flask import Blueprint, render_template, redirect, url_for, request, flash, jsonify
# from werkzeug.security import generate_password_hash, check_password_hash
# from flask_login import login_user, login_required, logout_user
# from models import db, User

# auth = Blueprint('auth', __name__)

# # Login Route
# @auth.route('/sign_in', methods=['POST', 'GET'])
# def login():
#     if request.method == 'POST':
#         # Handle JSON login
#         if request.is_json:
#             data = request.get_json()
#             email = data.get('email')
#             password = data.get('password')

#             if not email or not password:
#                 return jsonify({"error": "Email and password are required"}), 400

#             user = User.query.filter_by(email=email).first()
#             if user and check_password_hash(user.password, password):
#                 login_user(user)
#                 return jsonify({"message": "Login successful!"}), 200

#         # Handle form login
#         email = request.form.get("email")
#         password = request.form.get("password")
#         user = User.query.filter_by(email=email).first()

#         if user and check_password_hash(user.password, password):
#             login_user(user, remember=True)
#             flash('Login successful!', "success")
#             return redirect(url_for("welcome"))  # Redirect to the /welcome page after login
#         else:
#             flash("Login failed. Check your email & password.", 'danger')

#     return render_template('sign_in.html')


# # Registration Route
# @auth.route('/register', methods=["POST", "GET"])
# def register():
#     if request.method == "POST":
#         # Check if the request is in JSON format
#         if request.is_json:
#             data = request.get_json()  # Parse JSON data
#             name = data.get('name')
#             email = data.get('email')
#             password = data.get('password')

#             # Validate input
#             if not name or not email or not password:
#                 return jsonify({"error": "Name, email, and password are required"}), 400

#             # Check if the user already exists
#             existing_user = User.query.filter_by(email=email).first()
#             if existing_user:
#                 return jsonify({"error": "Email already registered. Please log in."}), 400

#             # Hash the password
#             hashed_password = generate_password_hash(password, method="pbkdf2:sha256")

#             # Create the new user
#             new_user = User(username=name, email=email, password=hashed_password)  # Fix 'name' to 'username'
#             db.session.add(new_user)
#             db.session.commit()

#             return jsonify({"message": "Registered successfully! Please log in."}), 201

#         # If it's not JSON, assume it's a form submission (handle the regular HTML form)
#         name = request.form.get('name')  # Get the 'name' field
#         email = request.form['email']
#         password = request.form["password"]

#         # Validate input
#         if not name or not email or not password:
#             flash("Name, email, and password are required.", "danger")
#             return redirect(url_for("auth.register"))

#         # Check if the user already exists
#         existing_user = User.query.filter_by(email=email).first()
#         if existing_user:
#             flash("Email already registered. Please log in.", "warning")
#             return redirect(url_for("auth.login"))

#         # Hash the password
#         hashed_password = generate_password_hash(password, method="pbkdf2:sha256")

#         # Create new user and store in the database
#         new_user = User(username=name, email=email, password=hashed_password)  # Fix 'name' to 'username'
#         db.session.add(new_user)
#         db.session.commit()

#         flash('Registered successfully! Please log in.', 'success')
#         return redirect(url_for("auth.login"))

#     return render_template('register.html')


# # Logout Route
# @auth.route('/logout')
# @login_required
# def logout():
#     logout_user()
#     flash('You have logged out successfully.', 'info')
#     return redirect(url_for('auth.login'))


from flask import Blueprint, render_template, redirect, url_for, request, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, login_required, logout_user, current_user
from models import db, User

auth = Blueprint('auth', __name__)

def authenticate_user(email, password):
    """Helper function to authenticate user."""
    user = User.query.filter_by(email=email).first()
    if user and check_password_hash(user.password, password):
        return user
    return None

# Login Route
@auth.route('/sign_in', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        # Handle JSON login
        if request.is_json:
            data = request.get_json()
            email = data.get('email')
            password = data.get('password')

            if not email or not password:
                return jsonify({"error": "Email and password are required"}), 400

            user = authenticate_user(email, password)
            if user:
                login_user(user)
                return jsonify({"message": "Login successful!"}), 200
            return jsonify({"error": "Invalid email or password."}), 400

        # Handle form login
        email = request.form.get("email")
        password = request.form.get("password")
        user = authenticate_user(email, password)

        if user:
            login_user(user, remember=True)
            flash('Login successful!', "success")
            return redirect(url_for("welcome"))  # Redirect to the /welcome page after login
        else:
            flash("Login failed. Check your email & password.", 'danger')

    return render_template('sign_in.html')

# Registration Route
@auth.route('/register', methods=["POST", "GET"])
def register():
    if request.method == "POST":
        # Check if the request is in JSON format
        if request.is_json:
            data = request.get_json()  # Parse JSON data
            name = data.get('name')
            email = data.get('email')
            password = data.get('password')

            # Validate input
            if not name or not email or not password:
                return jsonify({"error": "Name, email, and password are required"}), 400

            # Check if the user already exists
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                return jsonify({"error": "Email already registered. Please log in."}), 400

            # Hash the password
            hashed_password = generate_password_hash(password, method="pbkdf2:sha256")

            # Create the new user
            new_user = User(username=name, email=email, password=hashed_password)  
            db.session.add(new_user)
            db.session.commit()

            return jsonify({"message": "Registered successfully! Please log in."}), 201

        # If it's not JSON, assume it's a form submission (handle the regular HTML form)
        name = request.form.get('name')  # Get the 'name' field
        email = request.form['email']
        password = request.form["password"]

        # Validate input
        if not name or not email or not password:
            flash("Name, email, and password are required.", "danger")
            return redirect(url_for("auth.register"))

        # Check if the user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered. Please log in.", "warning")
            return redirect(url_for("auth.login"))

        # Hash the password
        hashed_password = generate_password_hash(password, method="pbkdf2:sha256")

        # Create new user and store in the database
        new_user = User(username=name, email=email, password=hashed_password)  
        db.session.add(new_user)
        db.session.commit()

        flash('Registered successfully! Please log in.', 'success')
        return redirect(url_for("auth.login"))

    return render_template('register.html')

# Logout Route
@auth.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have logged out successfully.', 'info')
    return redirect(url_for('auth.login'))
