from flask import Flask, request, jsonify
from sarvamai import SarvamAI
import os
import requests # Add this to download the audio file
import uuid     # Add this to create unique filenames

# Initialize the Flask app
app = Flask(__name__)

# Get your Sarvam AI API key from an environment variable
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")

# Check if the API key is set
if not SARVAM_API_KEY:
    raise ValueError("SARVAM_API_KEY environment variable not set.")

# Initialize the Sarvam AI client
client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

@app.route('/')
def home():
    # A simple route to check if the API is running
    return "Audio Analysis API is running!"

@app.route('/analyze_audio', methods=['POST'])
def analyze_audio():
    """
    This endpoint receives an audio URL, transcribes it using Sarvam.ai,
    and returns the transcription in JSON format.
    """
    data = request.get_json()
    if not data or 'audio_url' not in data:
        return jsonify({"error": "audio_url not provided"}), 400

    audio_url = data['audio_url']
    
    try:
        # --- Download the audio file from the URL ---
        response = requests.get(audio_url, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes

        # Create a unique filename to avoid conflicts
        temp_filename = f"{uuid.uuid4()}.wav" 

        with open(temp_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        # ---------------------------------------------

        job = client.speech_to_text_translate_job.create_job(
            model="saaras:v2.5",
            with_diarization=True
        )

        job.upload_files(file_paths=[temp_filename])
        job.start()
        job.wait_until_complete()

        if job.is_failed():
            os.remove(temp_filename) # Clean up the downloaded file
            return jsonify({"error": "Transcription failed"}), 500
        
        output_dir = f"outputs/transcriptions_{job.job_id}"
        os.makedirs(output_dir, exist_ok=True)
        job.download_outputs(output_dir=output_dir)

        json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
        if not json_files:
            os.remove(temp_filename) # Clean up
            return jsonify({"error": "No transcription file found"}), 500

        with open(os.path.join(output_dir, json_files[0]), 'r') as f:
            analysis = f.read()
        
        os.remove(temp_filename) # Clean up the downloaded file
        return analysis, 200

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to download audio file: {e}"}), 400
    except Exception as e:
        # Clean up the file if it exists
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return jsonify({"error": str(e)}), 500

# Note: We have removed the app.run() block
