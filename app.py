from flask import Flask, render_template, redirect, url_for, request, flash, send_file
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash #hashing passwords + verification
from pymongo import MongoClient #connect to MongoDB
from dotenv import load_dotenv #loading env variables from .env to os.enivorn
from io import BytesIO #pdf processing
import os #read env variables
import re #regular expressions

from bson import ObjectId
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq #AI model

# ---------- Load env ----------
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/ai_resume_db")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SECRET_KEY = os.getenv("FLASK_SECRET_KEY")

# ---------- Flask setup ----------
app = Flask(__name__, instance_relative_config=True)
app.config["SECRET_KEY"] = SECRET_KEY #reassures session cookies are signed and tamper-proof

# ---------- MongoDB ----------
mongo_client = MongoClient(MONGODB_URI) # connecting to MongoDB
db = mongo_client.get_default_database()  # if db in URI, else default to "ai_resume_db"
users_col = db.get_collection("users") # creates/gets the users collection for authentication

# ---------- Authentication model ----------
class User(UserMixin): #UserMixin class has necessary functions for authentication, class User is required for flask-login
    def __init__(self, user_doc): #getting user document from MongoDB
        self.id = str(user_doc["_id"])
        self.email = user_doc["email"]
        self.name = user_doc.get("name", self.email)

def find_user_by_email(email): #finds user by email in MongoDB
    return users_col.find_one({"email": email.lower().strip()})

def create_user(name, email, password): #creates user in MongoDB with hashed password
    hashed = generate_password_hash(password)
    users_col.insert_one({
        "name": name.strip(),
        "email": email.lower().strip(),
        "password_hash": hashed,
    })

# ---------- Login manager ----------
login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app) #mounting login manager to flask app

@login_manager.user_loader
def load_user(user_id): #since HTTP/S is stateless, this function reloads user object from user ID stored in session
    doc = users_col.find_one({"_id": ObjectId(user_id)})
    return User(doc) if doc else None

def is_strong_password(password: str) -> bool: #checks if password is strong
    if len(password) < 10: #minimum length 10
        return False
    if not re.search(r"[A-Z]", password): #force capital letter
        return False
    if not re.search(r"[a-z]", password): #force small letter
        return False
    if not re.search(r"[0-9]", password): #force digit
        return False
    if not re.search(r"[^A-Za-z0-9]", password): #force special character
        return False
    return True

# ---------- Core logic ----------
def extract_pdf_text(file_storage): #extracts text from uploaded PDF file
    
    try:
        file_storage.stream.seek(0)  # reset pointer to start of file
        pdf_bytes = file_storage.read() # read entire file into memory
        buf = BytesIO(pdf_bytes) # convert bytes to BytesIO object for pdfminer

        output_str = BytesIO() # output buffer for extracted text to be saved in memory
        laparams = LAParams() #pdfminer needs this object for turning positioned characters into words/lines
        extract_text_to_fp(buf, output_str, laparams=laparams, output_type='text', codec=None) #outputting text to output_str (encoded bytes)
        return output_str.getvalue().decode("utf-8", errors="ignore") # decode the encoded bytes to utf-8, ignoring errors

    except Exception as e:
        raise RuntimeError(f"Error extracting text from PDF: {e}")

# Cache model in memory so every request doesn’t reload it
_ats_model = None #Applicant Tracking System, used by recruiters to manage job applications
def get_sentence_model():
    global _ats_model #accessing the global variable
    if _ats_model is None:
        _ats_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    return _ats_model #sentenceTransformer is used to convert sentences to vectors to quantify similarity

def calculate_similarity_bert(text1, text2):
    model = get_sentence_model()
    embeddings1 = model.encode([text1])
    embeddings2 = model.encode([text2])
    similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
    return float(similarity)

def get_report(resume, job_desc):
    client = Groq(api_key=GROQ_API_KEY)
    prompt = f"""
    # Context:
    - You are an AI Resume Analyzer, you will be given Candidate's resume and Job Description of the role he is applying for.

    # Instruction:
    - Analyze candidate's resume based on the possible points that can be extracted from job description,and give your evaluation on each point with the criteria below:  
    - Consider all points like required skills, experience,etc that are needed for the job role.
    - Calculate the score to be given (out of 10) for every point based on evaluation at the beginning of each point with a detailed explanation.  
    - If the resume aligns with the job description point, mark it with ✅ and provide a detailed explanation.  
    - If the resume doesn't align with the job description point, mark it with ❌ and provide a reason for it.  
    - If a clear conclusion cannot be made, use a ⚠️ sign with a reason.  
    - The Final Heading should be "Suggestions to improve your resume:" and give where and what the candidate can improve to be selected for that job role.
    - If the job description is insufficient or unclear or only includes words not relating to a job, write a single line specifying that it's insufficient/unclear accordingly so 0/5, and don't recommend anything, can skip the final heading.
    
    # Inputs:
    Candidate Resume: {resume}
    ---
    Job Description: {job_desc}

    # Output:
    - Each any every point should be given a score (example: 3/10 ).
    - Mention the scores and relevant emoji at the beginning of each point and then explain the reason.
    """
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

def extract_scores(text):
    pattern = r'(\d+(?:\.\d+)?)/10'
    matches = re.findall(pattern, text)
    return [float(m) for m in matches]

# ---------- Routes ----------
@app.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("analyze"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "")
        password = request.form.get("password", "")
        user_doc = find_user_by_email(email) #checks if user exists in MongoDB
        if not user_doc or not check_password_hash(user_doc["password_hash"], password): #if email or password is incorrect, return to /login
            flash("Invalid email or password.", "danger")
            return redirect(url_for("login"))
        login_user(User(user_doc)) #creates user session
        flash("Welcome back!", "success")
        return redirect(url_for("analyze"))
    return render_template("login.html")

@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "POST": #if we want to post data to /signup
        name = request.form.get("name","").strip() #strip() removes leading/trailing whitespace
        email = request.form.get("email","").strip()
        password = request.form.get("password","")
        confirm = request.form.get("confirm","")

        if not name or not email or not password: #requires all fields to be filled, return to /signup
            flash("All fields are required.", "warning")
            return redirect(url_for("signup"))

        if password != confirm: #if passwords do not match when confirming password, return to /signup
            flash("Passwords do not match.", "warning")
            return redirect(url_for("signup"))

        if not is_strong_password(password): #if password policy is not met, return to /signup
            flash("Password must be at least 10 characters long and include uppercase, lowercase, digit, and symbol.", "danger")
            return redirect(url_for("signup"))

        if find_user_by_email(email): #if email already exists the database, return to /signup
            flash("An account with that email already exists.", "warning")
            return redirect(url_for("signup"))

        create_user(name, email, password) #creates mongoDB instance with registration user details, return to /login
        flash("Account created! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html") #if using http GET

@app.route("/logout")
@login_required
def logout(): #disconnects user session, return /login
    logout_user() #pops user from session array
    flash("Logged out.", "info")
    return redirect(url_for("login"))

@app.route("/analyze", methods=["GET","POST"])
@login_required
def analyze():
    if request.method == "POST":
        job_desc = request.form.get("job_desc","").strip()
        resume_file = request.files.get("resume_pdf")
        if not job_desc or not resume_file:
            flash("Please upload a PDF resume and enter a job description.", "warning")
            return redirect(url_for("analyze"))

        try:
            resume_text = extract_pdf_text(resume_file)
        except RuntimeError as e:
            flash(str(e), "danger")
            return redirect(url_for("analyze"))

        # Similarity (ATS-like)
        similarity_score = calculate_similarity_bert(resume_text, job_desc)

        # LLM Report
        report = get_report(resume_text, job_desc)
        scores = extract_scores(report)
        avg_score = round(sum(scores) / (10 * len(scores)), 4) if scores else 0.0

        # Keep values in session for the download endpoint
        request.session = {}  # simple container if needed
        # Alternatively, pass via querystring or server cache; we'll pass via render + hidden field
        return render_template(
            "result.html",
            similarity_score=similarity_score,
            avg_score=avg_score,
            report=report
        )
    return render_template("analyze.html")

@app.post("/download")
@login_required
def download():
    report_text = request.form.get("report_text","")
    if not report_text:
        flash("Nothing to download.", "warning")
        return redirect(url_for("analyze"))
    buf = BytesIO(report_text.encode("utf-8"))
    return send_file(
        buf,
        as_attachment=True,
        download_name="report.txt",
        mimetype="text/plain"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
