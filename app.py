from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import sqlite3
from datetime import datetime

from utils.image_detector import detect_deepfake
from utils.video_detector import extract_frames


app = Flask(__name__)
CORS(app)


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

FRAME_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "frames"
)

DATABASE = os.path.join(
    BASE_DIR,
    "deepguard.db"
)

ALLOWED_IMAGE_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}

ALLOWED_VIDEO_EXTENSIONS = {
    "mp4",
    "mov",
    "avi",
    "mkv",
    "webm"
}


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)


# =========================================================
# DATABASE
# =========================================================

def initialize_database():

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""
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


initialize_database()


# =========================================================
# SAVE HISTORY
# =========================================================

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

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO history
        (
            filename,
            file_type,
            result,
            confidence,
            real_probability,
            fake_probability,
            frames_analyzed,
            model,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        filename,
        file_type,
        result,
        confidence,
        real_probability,
        fake_probability,
        frames_analyzed,
        model,
        datetime.now()
    ))

    conn.commit()
    conn.close()


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template("index.html")


# =========================================================
# DETECTION PAGE
# =========================================================

@app.route("/detect")
def detect():

    return render_template("detect.html")


# =========================================================
# HISTORY PAGE
# =========================================================

@app.route("/history")
def history():

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM history
        ORDER BY created_at DESC
    """)

    records = cursor.fetchall()

    conn.close()

    total = len(records)

    fake_count = sum(
        1 for record in records
        if record["result"] == "FAKE"
    )

    real_count = sum(
        1 for record in records
        if record["result"] == "REAL"
    )

    return render_template(
        "history.html",
        records=records,
        total=total,
        fake_count=fake_count,
        real_count=real_count
    )


# =========================================================
# UPLOAD AND ANALYSIS
# =========================================================

@app.route("/upload", methods=["POST"])
def upload():

    if "file" not in request.files:

        return jsonify({
            "success": False,
            "error": "No file uploaded."
        }), 400


    file = request.files["file"]


    if file.filename == "":

        return jsonify({
            "success": False,
            "error": "No file selected."
        }), 400


    filename = file.filename

    extension = filename.rsplit(".", 1)[-1].lower()


    # =====================================================
    # IMAGE ANALYSIS
    # =====================================================

    if extension in ALLOWED_IMAGE_EXTENSIONS:

        file_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(file_path)


        try:

            analysis = detect_deepfake(
                file_path
            )

            result = analysis["result"]

            confidence = float(
                analysis["confidence"]
            )

            real_probability = float(
                analysis["real_probability"]
            )

            fake_probability = float(
                analysis["fake_probability"]
            )

            model = analysis["model"]

            explanation = analysis.get(
                "explanation",
                ""
            )


            save_history(
                filename,
                "Image",
                result,
                confidence,
                real_probability,
                fake_probability,
                0,
                model
            )


            result_url = (
                "/result?"
                f"filename={filename}"
                f"&result={result}"
                f"&confidence={confidence}"
                f"&real_probability={real_probability}"
                f"&fake_probability={fake_probability}"
                f"&frames_analyzed=0"
                f"&model={model}"
                f"&analysis_type=Image%20Analysis"
                f"&fake_frames=0"
                f"&real_frames=0"
                f"&explanation={explanation}"
            )


            return jsonify({
                "success": True,
                "result_url": result_url
            })


        except Exception as e:

            print("Image analysis error:", e)

            return jsonify({
                "success": False,
                "error": str(e)
            }), 500


    # =====================================================
    # VIDEO ANALYSIS
    # =====================================================

    elif extension in ALLOWED_VIDEO_EXTENSIONS:

        file_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(file_path)


        try:

            print("\n==============================")
            print("Starting DeepGuard Video Analysis")
            print("==============================\n")


            # ---------------------------------------------
            # Extract frames
            # ---------------------------------------------

            frames = extract_frames(
                file_path
            )


            if not frames:

                return jsonify({
                    "success": False,
                    "error": "Unable to extract frames from the video."
                }), 500


            print(
                f"Frames extracted: {len(frames)}"
            )


            # ---------------------------------------------
            # Analyze every frame
            # ---------------------------------------------

            frame_results = []

            total_real_probability = 0.0
            total_fake_probability = 0.0

            fake_frames = 0
            real_frames = 0


            for index, frame_path in enumerate(frames):

                print(
                    f"Analyzing frame "
                    f"{index + 1}/{len(frames)}..."
                )


                analysis = detect_deepfake(
                    frame_path
                )


                frame_results.append(
                    analysis
                )


                total_real_probability += float(
                    analysis["real_probability"]
                )

                total_fake_probability += float(
                    analysis["fake_probability"]
                )


                if analysis["result"] == "FAKE":

                    fake_frames += 1

                else:

                    real_frames += 1


            # ---------------------------------------------
            # Calculate average probabilities
            # ---------------------------------------------

            frames_analyzed = len(
                frame_results
            )


            average_real_probability = (
                total_real_probability /
                frames_analyzed
            )


            average_fake_probability = (
                total_fake_probability /
                frames_analyzed
            )


            # ---------------------------------------------
            # Final video result
            # ---------------------------------------------

            if (
                average_fake_probability >=
                average_real_probability
            ):

                final_result = "FAKE"

                final_confidence = (
                    average_fake_probability
                )

                explanation = (
                    f"DeepGuard analyzed "
                    f"{frames_analyzed} video frames. "
                    f"{fake_frames} frames were classified "
                    f"as AI-generated or manipulated, while "
                    f"{real_frames} frames were classified "
                    f"as real-looking."
                )

            else:

                final_result = "REAL"

                final_confidence = (
                    average_real_probability
                )

                explanation = (
                    f"DeepGuard analyzed "
                    f"{frames_analyzed} video frames. "
                    f"{real_frames} frames were classified "
                    f"as real-looking, while "
                    f"{fake_frames} frames were classified "
                    f"as AI-generated or manipulated."
                )


            model = frame_results[0]["model"]


            # ---------------------------------------------
            # Save video history
            # ---------------------------------------------

            save_history(
                filename,
                "Video",
                final_result,
                round(final_confidence, 2),
                round(average_real_probability, 2),
                round(average_fake_probability, 2),
                frames_analyzed,
                model
            )


            # ---------------------------------------------
            # Result page
            # ---------------------------------------------

            result_url = (
                "/result?"
                f"filename={filename}"
                f"&result={final_result}"
                f"&confidence={round(final_confidence, 2)}"
                f"&real_probability={round(average_real_probability, 2)}"
                f"&fake_probability={round(average_fake_probability, 2)}"
                f"&frames_analyzed={frames_analyzed}"
                f"&model={model}"
                f"&analysis_type=Video%20Analysis"
                f"&fake_frames={fake_frames}"
                f"&real_frames={real_frames}"
                f"&explanation={explanation}"
            )


            return jsonify({
                "success": True,
                "result_url": result_url
            })


        except Exception as e:

            print(
                "Video analysis error:",
                e
            )

            return jsonify({
                "success": False,
                "error": str(e)
            }), 500


    # =====================================================
    # INVALID FILE
    # =====================================================

    else:

        return jsonify({
            "success": False,
            "error": (
                "Unsupported file type. "
                "Please upload an image or video."
            )
        }), 400


# =========================================================
# RESULT PAGE
# =========================================================

@app.route("/result")
def result():

    filename = request.args.get(
        "filename",
        ""
    )

    result_value = request.args.get(
        "result",
        "UNKNOWN"
    )

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

    frames_analyzed = request.args.get(
        "frames_analyzed",
        "0"
    )

    model = request.args.get(
        "model",
        "DeepGuard AI Model"
    )

    analysis_type = request.args.get(
        "analysis_type",
        "Media Analysis"
    )

    fake_frames = request.args.get(
        "fake_frames",
        "0"
    )

    real_frames = request.args.get(
        "real_frames",
        "0"
    )

    explanation = request.args.get(
        "explanation",
        ""
    )


    return render_template(
        "result.html",

        filename=filename,

        result=result_value,

        confidence=confidence,

        real_probability=real_probability,

        fake_probability=fake_probability,

        frames_analyzed=frames_analyzed,

        model=model,

        analysis_type=analysis_type,

        fake_frames=fake_frames,

        real_frames=real_frames,

        explanation=explanation
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    print("\n===================================")
    print("       DEEPGUARD AI DETECTOR")
    print("===================================")
    print("Server: http://127.0.0.1:5000")
    print("===================================\n")

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )