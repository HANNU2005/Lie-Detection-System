import cv2
import numpy as np
import streamlit as st
import librosa
import pyaudio
import wave
import threading
import time
from scipy.signal import find_peaks
import tempfile
import os
import atexit
import sounddevice as sd

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #FFFFFF;
        color: #333333;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .result-box {
        background-color: #F5F5F5;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-box {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
        border: 1px solid #E0E0E0;
    }
    h1 {
        color: #4CAF50;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 30px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    h2 {
        color: #4CAF50;
        font-size: 1.8em;
        margin-top: 20px;
    }
    .truth {
        color: #4CAF50;
        font-weight: bold;
    }
    .lie {
        color: #f44336;
        font-weight: bold;
    }
    .video-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

class LieDetector:
    def __init__(self):
        # Initialize OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        self.recording = False
        self.audio_frames = []
        self.video_frames = []
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.audio = pyaudio.PyAudio()
        self.cap = None
        self.audio_thread = None
        self.stream = None
        
        # Register cleanup
        atexit.register(self.cleanup)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio is not None:
            self.audio.terminate()
        cv2.destroyAllWindows()
    
    def get_face_features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
            
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(face_roi)
        eye_centers = []
        for (ex, ey, ew, eh) in eyes:
            eye_centers.append((x + ex + ew/2, y + ey + eh/2))
            
        # Detect smile
        smile = self.smile_cascade.detectMultiScale(face_roi, 1.7, 20)
        smile_height = 0
        if len(smile) > 0:
            (sx, sy, sw, sh) = smile[0]
            smile_height = sh
            
        # Calculate head tilt using eye centers
        head_tilt = 0
        if len(eye_centers) == 2:
            dx = eye_centers[1][0] - eye_centers[0][0]
            dy = eye_centers[1][1] - eye_centers[0][1]
            head_tilt = abs(np.arctan2(dy, dx) * 180. / np.pi)
            
        # Calculate looking direction
        face_center_x = x + w/2
        frame_center_x = frame.shape[1] / 2
        looking_direction = "center"
        if abs(face_center_x - frame_center_x) > frame.shape[1] * 0.1:
            looking_direction = "left" if face_center_x < frame_center_x else "right"
            
        return {
            'face_rect': (x, y, w, h),
            'eye_centers': eye_centers,
            'smile_height': smile_height,
            'head_tilt': head_tilt,
            'looking_direction': looking_direction
        }

    def analyze_facial_expression(self, frame):
        features = self.get_face_features(frame)
        if features is None:
            return None
            
        # Calculate metrics
        eye_aspect_ratio = 1.0  # Default value
        if len(features['eye_centers']) == 2:
            # Simple eye aspect ratio based on distance between eyes
            eye_distance = np.linalg.norm(
                np.array(features['eye_centers'][0]) - 
                np.array(features['eye_centers'][1])
            )
            eye_aspect_ratio = eye_distance / features['face_rect'][2]
            
        return {
            'eye_aspect_ratio': eye_aspect_ratio,
            'mouth_height': features['smile_height'] / features['face_rect'][3],
            'head_tilt': features['head_tilt'],
            'looking_direction': features['looking_direction']
        }
    
    def save_audio_to_wav(self, audio_frames):
        """Save recorded audio to a temporary WAV file"""
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        with wave.open(temp_audio.name, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(audio_frames))
        return temp_audio.name

    def play_recorded_video(self, frames, audio_file, placeholder):
        """Play the recorded video frames with audio"""
        # Start audio playback with better quality settings
        audio_data, sr = librosa.load(audio_file, sr=self.RATE, mono=True)
        
        # Normalize audio to prevent clipping
        audio_data = librosa.util.normalize(audio_data)
        
        # Play audio with better quality
        sd.play(audio_data, sr)
        
        # Play video frames with faster response
        for frame in frames:
            placeholder.image(frame, channels="BGR", use_container_width=True)
            time.sleep(0.02)  # Reduced from 0.03 to 0.02 for faster response
        
        # Wait for audio to finish
        sd.wait()
        placeholder.empty()
        
        # Clean up temporary audio file
        try:
            os.unlink(audio_file)
        except:
            pass

    def analyze_voice(self, audio_data):
        # Convert audio data to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.float32)
        
        # Check for and handle non-finite values
        if not np.all(np.isfinite(audio_array)):
            # Replace non-finite values with zeros
            audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize audio data
        audio_array = librosa.util.normalize(audio_array)
        
        # Calculate basic pitch metrics
        pitches, magnitudes = librosa.piptrack(y=audio_array, sr=self.RATE)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        
        if len(pitch_values) == 0:
            return None
        
        # Simple pitch analysis
        mean_pitch = np.mean(pitch_values)
        pitch_std = np.std(pitch_values)
        
        return {
            'mean_pitch': mean_pitch,
            'pitch_variation': pitch_std
        }
    
    def start_recording(self):
        self.recording = True
        self.audio_frames = []
        self.video_frames = []
        
        def record_audio():
            try:
                self.stream = self.audio.open(
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.RATE,
                    input=True,
                    frames_per_buffer=self.CHUNK,
                    input_device_index=None  # Use default input device
                )
                
                while self.recording:
                    data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                    self.audio_frames.append(data)
            except Exception as e:
                print(f"Audio recording error: {e}")
            finally:
                if self.stream is not None:
                    self.stream.stop_stream()
                    self.stream.close()
                    self.stream = None
        
        self.audio_thread = threading.Thread(target=record_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()
    
    def stop_recording(self):
        self.recording = False
        if self.audio_thread is not None:
            self.audio_thread.join(timeout=2.0)
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    def analyze_results(self, facial_data, voice_data):
        if not facial_data:
            return "Insufficient data for analysis"
        
        # Initialize truth and lie indicators
        truth_indicators = 0
        lie_indicators = 0
        
        # Face alignment analysis - more sensitive
        if facial_data['looking_direction'] == "center":
            truth_indicators += 1
        else:
            lie_indicators += 2  # Increased penalty for not looking center
        
        # Head tilt analysis - more sensitive
        if facial_data['head_tilt'] < 5:  # Reduced from 10 to 5 degrees
            truth_indicators += 1
        else:
            lie_indicators += 1
        
        # Eye aspect ratio analysis - more sensitive
        if 0.25 <= facial_data['eye_aspect_ratio'] <= 0.35:  # Narrower range
            truth_indicators += 1
        else:
            lie_indicators += 1
        
        # Mouth movement analysis - more sensitive
        if facial_data['mouth_height'] < 0.2:  # Reduced from 0.3 to 0.2
            truth_indicators += 1
        else:
            lie_indicators += 1
        
        # Calculate final score
        total_indicators = truth_indicators + lie_indicators
        if total_indicators == 0:
            return "Insufficient data for analysis"
        
        truth_percentage = (truth_indicators / total_indicators) * 100
        
        # Determine result with adjusted thresholds
        if truth_percentage >= 80:  # Increased from 75
            return "Truth"
        elif truth_percentage >= 60:  # Changed from 50
            return "Truth..."
        elif truth_percentage >= 40:  # Changed from 25
            return "Likely Lie"
        else:
            return "Definite Lie"

def play_lie_alert():
    """Play a beep sound for lie detection"""
    # Generate a beep sound
    sample_rate = 44100  # 44.1kHz
    duration = 0.7  # 500ms
    frequency = 1000  # 1kHz
    
    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    note = np.sin(frequency * t * 2 * np.pi)
    
    # Play the sound
    sd.play(note, sample_rate)
    sd.wait()  # Wait until sound is finished playing

def main():
    st.markdown("<h1>Lie Detection System</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <p style='font-size: 1.2em; color: #666666;'>
            Advanced facial expression analysis for deception detection
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'lie_detector' not in st.session_state:
        st.session_state.lie_detector = LieDetector()
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'video_frames' not in st.session_state:
        st.session_state.video_frames = []
    if 'audio_frames' not in st.session_state:
        st.session_state.audio_frames = []
    if 'show_recorded' not in st.session_state:
        st.session_state.show_recorded = False
    
    # Create columns for the interface
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_button = st.button("🎥 Start Recording")
    with col2:
        stop_button = st.button("⏹️ Stop Recording")
    with col3:
        analyze_button = st.button("🔍 Analyze")
    
    # Create a placeholder for the video feed
    video_placeholder = st.empty()
    
    # Handle button clicks
    if start_button and not st.session_state.recording:
        st.session_state.recording = True
        st.session_state.lie_detector.start_recording()
        st.session_state.video_frames = []
        st.session_state.audio_frames = []
        st.session_state.show_recorded = False
        st.markdown("<div class='result-box'>🎥 Recording started...</div>", unsafe_allow_html=True)
        
        # Start webcam
        st.session_state.lie_detector.cap = cv2.VideoCapture(0)
        
        try:
            while st.session_state.recording:
                ret, frame = st.session_state.lie_detector.cap.read()
                if not ret:
                    break
                
                # Store frame for analysis
                st.session_state.video_frames.append(frame)
                
                # Display webcam feed directly without container
                video_placeholder.image(frame, channels="BGR", use_container_width=True)
                
                # Add a small delay to prevent overwhelming the system
                time.sleep(0.03)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            if st.session_state.lie_detector.cap is not None:
                st.session_state.lie_detector.cap.release()
                st.session_state.lie_detector.cap = None
    
    if stop_button and st.session_state.recording:
        st.session_state.recording = False
        st.session_state.lie_detector.stop_recording()
        st.session_state.audio_frames = st.session_state.lie_detector.audio_frames
        st.markdown("<div class='result-box'>⏹️ Recording stopped.</div>", unsafe_allow_html=True)
        
        # Show recorded video with audio
        if st.session_state.video_frames:
            st.markdown("<div class='result-box'>🎬 Playing recorded video with audio...</div>", unsafe_allow_html=True)
            # Save audio to temporary file
            audio_file = st.session_state.lie_detector.save_audio_to_wav(st.session_state.audio_frames)
            # Play video with audio
            st.session_state.lie_detector.play_recorded_video(st.session_state.video_frames, audio_file, video_placeholder)
            st.session_state.show_recorded = True
    
    if analyze_button and not st.session_state.recording:
        if not st.session_state.video_frames or not st.session_state.audio_frames:
            st.markdown("<div class='result-box' style='color: #f44336;'>❌ No data to analyze. Please record first.</div>", unsafe_allow_html=True)
        else:
            with st.spinner('🔍 Analyzing...'):
                # Analyze video frames
                facial_data_list = []
                for frame in st.session_state.video_frames:
                    facial_data = st.session_state.lie_detector.analyze_facial_expression(frame)
                    if facial_data:
                        facial_data_list.append(facial_data)
                
                if not facial_data_list:
                    st.markdown("<div class='result-box' style='color: #f44336;'>❌ No facial data detected in the recording.</div>", unsafe_allow_html=True)
                    return
                
                # Analyze voice data
                voice_data = st.session_state.lie_detector.analyze_voice(b''.join(st.session_state.audio_frames))
                
                # Get average facial data
                avg_facial_data = {
                    'eye_aspect_ratio': np.mean([d['eye_aspect_ratio'] for d in facial_data_list]),
                    'mouth_height': np.mean([d['mouth_height'] for d in facial_data_list]),
                    'head_tilt': np.mean([d['head_tilt'] for d in facial_data_list]),
                    'looking_direction': max(set([d['looking_direction'] for d in facial_data_list]), 
                                          key=[d['looking_direction'] for d in facial_data_list].count)
                }
                
                # Get analysis result
                result = st.session_state.lie_detector.analyze_results(avg_facial_data, voice_data)
                
                # Play beep sound for lie detection
                if result in ["Likely Lie", "Definite Lie"]:
                    play_lie_alert()
                
                # Display results with custom styling
                st.markdown("<h2>Analysis Results</h2>", unsafe_allow_html=True)
                result_class = "truth" if result == "Truth" else "lie"
                st.markdown(f"<div class='result-box'><h3 class='{result_class}'>Result: {result}</h3></div>", unsafe_allow_html=True)
                
                # Display detailed metrics
                st.markdown("<h2>Detailed Metrics</h2>", unsafe_allow_html=True)
                st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
                st.markdown("<h3>Facial Analysis:</h3>", unsafe_allow_html=True)
                st.markdown(f"<p>👁️ Eye Aspect Ratio: {avg_facial_data['eye_aspect_ratio']:.3f}</p>", unsafe_allow_html=True)
                st.markdown(f"<p>👄 Mouth Height: {avg_facial_data['mouth_height']:.3f}</p>", unsafe_allow_html=True)
                st.markdown(f"<p>🔄 Head Tilt: {avg_facial_data['head_tilt']:.3f}</p>", unsafe_allow_html=True)
                st.markdown(f"<p>👀 Looking Direction: {avg_facial_data['looking_direction']}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                if voice_data:
                    st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
                    st.markdown("<h3>Voice Analysis:</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p>🎵 Mean Pitch: {voice_data['mean_pitch']:.2f}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p>📊 Pitch Variation: {voice_data['pitch_variation']:.2f}</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main() 