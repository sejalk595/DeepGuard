import os
import sqlite3
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from utils.image_detector import detect_deepfake
from utils.video_detector import extract_frames


# ============================================================
# DEEPGUARD - AI GENERATED MEDIA DETECTION SYSTEM
# ============================================================

app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
FRAMES_FOLDER = os.path.join(BASE_DIR, "static", "frames")
DATABASE = os.path.join(BASE_DIR, "deepguard.db")

ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB


# ------------------------------------------------------------
# Create required folders
# ------------------------------------------------------------

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)


# ============================================================
# DATABASE
# ============================================================

def initialize_database():
    conn = sqlite3.connect(DATABASE)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            result TEXT NOT NULL,
            confidence REAL NOT NULL,
            real_probability REAL NOT NULL,
            fake_probability REAL NOT NULL,
            frames_analyzed INTEGER DEFAULT 0,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def save_history(
    filename,
    file_type,
    result,
    confidence,
    real_probability,
    fake_probability,
    frames_analyzed,
    model
):
    conn = sqlite3.connect(DATABASE)

    conn.execute("""
        INSERT INTO history (
            filename,
            file_type,
            result,
            confidence,
            real_probability,
            fake_probability,
            frames_analyzed,
            model
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        filename,
        file_type,
        result,
        confidence,
        real_probability,
        fake_probability,
        frames_analyzed,
        model
    ))

    conn.commit()
    conn.close()


def get_history():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row

    records = conn.execute("""
        SELECT *
        FROM history
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    return records


initialize_database()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def allowed_file(filename):
    if not filename or "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()

    return (
        extension in ALLOWED_IMAGE_EXTENSIONS
        or extension in ALLOWED_VIDEO_EXTENSIONS
    )


def get_file_type(filename):
    extension = filename.rsplit(".", 1)[1].lower()

    if extension in ALLOWED_IMAGE_EXTENSIONS:
        return "Image"

    if extension in ALLOWED_VIDEO_EXTENSIONS:
        return "Video"

    return "Unknown"


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# DETECTION PAGE
# ============================================================

@app.route("/detect")
def detect_page():
    return render_template("detect.html")


# ============================================================
# HISTORY PAGE
# ============================================================

@app.route("/history")
def history():
    records = get_history()

    total_analyses = len(records)

    fake_detected = sum(
        1 for record in records
        if record["result"].upper() == "FAKE"
    )

    real_detected = sum(
        1 for record in records
        if record["result"].upper() == "REAL"
    )

    return render_template(
        "history.html",
        records=records,
        total_analyses=total_analyses,
        fake_detected=fake_detected,
        real_detected=real_detected
    )


# ============================================================
# UPLOAD AND ANALYSIS
# ============================================================

@app.route("/upload", methods=["POST"])
def upload_file():

    if "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "No file was uploaded."
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "No file was selected."
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Unsupported file type."
        }), 400

    # Secure filename
    filename = secure_filename(file.filename)

    # Prevent filename conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name, extension = os.path.splitext(filename)

    filename = f"{name}_{timestamp}{extension}"

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    # Save uploaded file
    file.save(filepath)

    file_type = get_file_type(filename)

    # ========================================================
    # IMAGE ANALYSIS
    # ========================================================

    if file_type == "Image":

        try:
            result_data = detect_deepfake(filepath)

            result = result_data["result"]
            confidence = float(result_data["confidence"])

            real_probability = float(
                result_data["real_probability"]
            )

            fake_probability = float(
                result_data["fake_probability"]
            )

            model = result_data.get(
                "model",
                "delpot/steganograph-ia-detector"
            )

            explanation = result_data.get(
                "explanation",
                "The image was analyzed using an AI-generated image detection model."
            )

            # Save to history
            save_history(
                filename=filename,
                file_type="Image",
                result=result,
                confidence=confidence,
                real_probability=real_probability,
                fake_probability=fake_probability,
                frames_analyzed=0,
                model=model
            )

            result_url = (
                "/result"
                f"?filename={filename}"
                f"&result={result}"
                f"&confidence={confidence}"
                f"&real_probability={real_probability}"
                f"&fake_probability={fake_probability}"
                f"&analysis_type=Image"
                f"&frames_analyzed=0"
                f"&fake_frames=0"
                f"&real_frames=0"
                f"&model={model}"
                f"&explanation={explanation}"
            )

            return jsonify({
                "success": True,
                "result_url": result_url
            })

        except Exception as error:

            print("IMAGE ANALYSIS ERROR:", error)

            return jsonify({
                "success": False,
                "error": f"Image analysis failed: {str(error)}"
            }), 500

    # ========================================================
    # VIDEO ANALYSIS
    # ========================================================

    if file_type == "Video":

        try:
            print("\n==============================")
            print("VIDEO ANALYSIS STARTED")
            print("==============================")

            # Extract frames
            frame_paths = extract_frames(filepath)

            if not frame_paths:

                return jsonify({
                    "success": False,
                    "error": "Unable to extract frames from the video."
                }), 400

            print(
                f"Frames extracted: {len(frame_paths)}"
            )

            real_probabilities = []
            fake_probabilities = []

            fake_frames = 0
            real_frames = 0

            model = "delpot/steganograph-ia-detector"

            # ------------------------------------------------
            # Analyze every extracted frame
            # ------------------------------------------------

            for index, frame_path in enumerate(frame_paths):

                try:

                    print(
                        f"Analyzing frame "
                        f"{index + 1}/{len(frame_paths)}..."
                    )

                    frame_result = detect_deepfake(
                        frame_path
                    )

                    frame_real = float(
                        frame_result["real_probability"]
                    )

                    frame_fake = float(
                        frame_result["fake_probability"]
                    )

                    real_probabilities.append(frame_real)
                    fake_probabilities.append(frame_fake)

                    model = frame_result.get(
                        "model",
                        model
                    )

                    if frame_result["result"].upper() == "FAKE":
                        fake_frames += 1
                    else:
                        real_frames += 1

                except Exception as frame_error:

                    print(
                        f"Frame {index + 1} failed:",
                        frame_error
                    )

            # ------------------------------------------------
            # Make sure at least one frame was analyzed
            # ------------------------------------------------

            if not real_probabilities:

                return jsonify({
                    "success": False,
                    "error": "No video frames could be analyzed."
                }), 500

            # ------------------------------------------------
            # Average frame probabilities
            # ------------------------------------------------

            average_real = (
                sum(real_probabilities)
                / len(real_probabilities)
            )

            average_fake = (
                sum(fake_probabilities)
                / len(fake_probabilities)
            )

            # ------------------------------------------------
            # Final video result
            # ------------------------------------------------

            if average_fake >= average_real:
                result = "FAKE"
                confidence = average_fake
            else:
                result = "REAL"
                confidence = average_real

            confidence = round(confidence, 2)

            average_real = round(average_real, 2)
            average_fake = round(average_fake, 2)

            frames_analyzed = len(real_probabilities)

            explanation = (
                f"The video was analyzed frame-by-frame. "
                f"{fake_frames} frames were classified as FAKE "
                f"and {real_frames} frames were classified as REAL."
            )

            print("\n==============================")
            print("VIDEO ANALYSIS COMPLETE")
            print("==============================")

            print("Result:", result)
            print("Confidence:", confidence)
            print("Real probability:", average_real)
            print("Fake probability:", average_fake)
            print("Frames analyzed:", frames_analyzed)

            # Save to history
            save_history(
                filename=filename,
                file_type="Video",
                result=result,
                confidence=confidence,
                real_probability=average_real,
                fake_probability=average_fake,
                frames_analyzed=frames_analyzed,
                model=model
            )

            result_url = (
                "/result"
                f"?filename={filename}"
                f"&result={result}"
                f"&confidence={confidence}"
                f"&real_probability={average_real}"
                f"&fake_probability={average_fake}"
                f"&analysis_type=Video"
                f"&frames_analyzed={frames_analyzed}"
                f"&fake_frames={fake_frames}"
                f"&real_frames={real_frames}"
                f"&model={model}"
                f"&explanation={explanation}"
            )

            return jsonify({
                "success": True,
                "result_url": result_url
            })

        except Exception as error:

            print("VIDEO ANALYSIS ERROR:", error)

            return jsonify({
                "success": False,
                "error": f"Video analysis failed: {str(error)}"
            }), 500

    return jsonify({
        "success": False,
        "error": "Unable to determine file type."
    }), 400


# ============================================================
# RESULT PAGE
# ============================================================

@app.route("/result")
def result():

    filename = request.args.get("filename", "")
    result_value = request.args.get("result", "UNKNOWN")

    confidence = request.args.get(
        "confidence",
        "0"
    )

    real_probability = request.args.get(
        "real_probability",
        "0"
    )

    fake_probability = request.args.get(
        "fake_probability",
        "0"
    )

    analysis_type = request.args.get(
        "analysis_type",
        "Image"
    )

    frames_analyzed = request.args.get(
        "frames_analyzed",
        "0"
    )

    fake_frames = request.args.get(
        "fake_frames",
        "0"
    )

    real_frames = request.args.get(
        "real_frames",
        "0"
    )

    model = request.args.get(
        "model",
        "delpot/steganograph-ia-detector"
    )

    explanation = request.args.get(
        "explanation",
        "The media was analyzed using an AI-generated media detection model."
    )

    return render_template(
        "result.html",
        filename=filename,
        result=result_value,
        confidence=confidence,
        real_probability=real_probability,
        fake_probability=fake_probability,
        analysis_type=analysis_type,
        frames_analyzed=frames_analyzed,
        fake_frames=fake_frames,
        real_frames=real_frames,
        model=model,
        explanation=explanation
    )


# ============================================================
# HEALTH CHECK
# Useful for cloud deployment
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "application": "DeepGuard",
        "message": "DeepGuard server is running."
    })


# ============================================================
# ERROR HANDLING
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "success": False,
        "error": "File is too large. Maximum size is 200 MB."
    }), 413


@app.errorhandler(500)
def internal_server_error(error):

    return jsonify({
        "success": False,
        "error": "An internal server error occurred."
    }), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )