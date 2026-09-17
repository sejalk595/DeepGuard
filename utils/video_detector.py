import os
import subprocess
from imageio_ffmpeg import get_ffmpeg_exe


# =========================================================
# EXTRACT VIDEO FRAMES
# =========================================================

def extract_frames(
    video_path,
    output_folder="static/frames",
    max_frames=20
):

    print("\n==========================================")
    print("VIDEO FRAME EXTRACTION")
    print("==========================================")

    if not os.path.exists(
        video_path
    ):

        raise FileNotFoundError(
            "Video file was not found."
        )

    # -----------------------------------------------------
    # CREATE / CLEAN FRAME DIRECTORY
    # -----------------------------------------------------

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    for filename in os.listdir(
        output_folder
    ):

        path = os.path.join(
            output_folder,
            filename
        )

        if os.path.isfile(path):

            os.remove(path)

    # -----------------------------------------------------
    # GET FFMPEG
    # -----------------------------------------------------

    ffmpeg = get_ffmpeg_exe()

    print(
        "FFmpeg ready."
    )

    # -----------------------------------------------------
    # EXTRACT 1 FRAME / SECOND
    # -----------------------------------------------------

    output_pattern = os.path.join(
        output_folder,
        "frame_%03d.jpg"
    )

    command = [

        ffmpeg,

        "-i",
        video_path,

        "-vf",
        "fps=1",

        "-frames:v",
        str(max_frames),

        "-q:v",
        "2",

        "-y",

        output_pattern
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:

        print(
            result.stderr
        )

        raise ValueError(
            "FFmpeg could not process "
            "the video."
        )

    # -----------------------------------------------------
    # COLLECT FRAMES
    # -----------------------------------------------------

    frames = []

    for filename in sorted(
        os.listdir(
            output_folder
        )
    ):

        if filename.lower().endswith(
            ".jpg"
        ):

            frames.append(
                os.path.join(
                    output_folder,
                    filename
                )
            )

    if not frames:

        raise ValueError(
            "No frames could be extracted."
        )

    print(
        "Frames extracted:",
        len(frames)
    )

    return frames