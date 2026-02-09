"""
Cogno Solution - Backend API Server
====================================
Flask backend for AI processing, camera analysis, and server-side operations.
Designed for deployment on Render.

Features:
- Dyspraxia movement analysis (OpenCV + MediaPipe)
- Text simplification (Transformers)
- Handwriting analysis
- Email notifications
- Report generation
- Supabase integration
"""

import os
import io
import json
import base64
import logging
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# CORS configuration - allow frontend origins
CORS(app, origins=[
    'http://localhost:3000',
    'http://localhost:5000',
    'http://localhost:5500',
    'http://localhost:5501',
    'http://127.0.0.1:3000',
    'http://127.0.0.1:5000',
    'http://127.0.0.1:5500',
    'http://127.0.0.1:5501',
    'http://localhost:*',
    'http://127.0.0.1:*',
    os.getenv('FRONTEND_URL', '*'),
    '*'  # Allow all origins during development
])

# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Application configuration."""
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')
    SMTP_HOST = os.getenv('SMTP_HOST', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
    SMTP_USER = os.getenv('SMTP_USER')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
    EMAIL_FROM = os.getenv('EMAIL_FROM', 'noreply@cognosolution.com')
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')

# =============================================================================
# Supabase Client
# =============================================================================

try:
    from supabase import create_client, Client
    supabase: Optional[Client] = None
    
    if Config.SUPABASE_URL and Config.SUPABASE_SERVICE_KEY:
        supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_SERVICE_KEY)
        logger.info("Supabase client initialized successfully")
    else:
        logger.warning("Supabase credentials not configured")
except ImportError:
    supabase = None
    logger.warning("Supabase library not installed")

# =============================================================================
# MediaPipe for Movement Analysis
# =============================================================================

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_AVAILABLE = True
    logger.info("MediaPipe initialized successfully")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not installed - dyspraxia features limited")

# =============================================================================
# Transformers for Text Processing
# =============================================================================

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    
    # Load text simplification model (lazy loading)
    _text_simplifier = None
    _summarizer = None
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers library available")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not installed - text processing features limited")

def get_text_simplifier():
    """Lazy load text simplification model."""
    global _text_simplifier
    if not TRANSFORMERS_AVAILABLE:
        return None
    if _text_simplifier is None:
        try:
            _text_simplifier = pipeline(
                "text2text-generation",
                model="facebook/bart-large-cnn",
                max_length=150
            )
            logger.info("Text simplifier model loaded")
        except Exception as e:
            logger.error(f"Failed to load text simplifier: {e}")
    return _text_simplifier

def get_summarizer():
    """Lazy load summarization model."""
    global _summarizer
    if not TRANSFORMERS_AVAILABLE:
        return None
    if _summarizer is None:
        try:
            _summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            logger.info("Summarizer model loaded")
        except Exception as e:
            logger.error(f"Failed to load summarizer: {e}")
    return _summarizer

# =============================================================================
# Authentication Middleware
# =============================================================================

def require_auth(f):
    """Decorator to require Supabase authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        
        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing authorization token'}), 401
        
        token = auth_header.split(' ')[1]
        
        if supabase:
            try:
                # Verify token with Supabase
                user = supabase.auth.get_user(token)
                if not user:
                    return jsonify({'error': 'Invalid token'}), 401
                request.user = user.user
            except Exception as e:
                logger.error(f"Auth verification failed: {e}")
                return jsonify({'error': 'Authentication failed'}), 401
        else:
            # Development mode - skip auth
            request.user = {'id': 'dev-user'}
        
        return f(*args, **kwargs)
    return decorated

def optional_auth(f):
    """Decorator for optional authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        request.user = None
        
        if auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            if supabase:
                try:
                    user = supabase.auth.get_user(token)
                    if user:
                        request.user = user.user
                except Exception:
                    pass
        
        return f(*args, **kwargs)
    return decorated

# =============================================================================
# Health Check Endpoints
# =============================================================================

@app.route('/', methods=['GET'])
def index():
    """Root endpoint."""
    return jsonify({
        'name': 'Cogno Solution API',
        'version': '1.0.0',
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    health = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'services': {
            'supabase': supabase is not None,
            'mediapipe': MEDIAPIPE_AVAILABLE,
            'transformers': TRANSFORMERS_AVAILABLE
        }
    }
    return jsonify(health)

# =============================================================================
# Dyspraxia - Movement Analysis Endpoints
# =============================================================================

@app.route('/api/dyspraxia/analyze-frame', methods=['POST'])
@optional_auth
def analyze_frame():
    """
    Analyze a single camera frame for movement/pose detection.
    
    Request:
        - frame or frame_data: Base64 encoded image or file upload
        - exercise_type: Type of exercise being performed
    
    Response:
        - landmarks: Detected body/hand landmarks
        - score: Movement accuracy score (0-100)
        - feedback: Real-time feedback message
        - frame_received: Debug flag (True when frame is successfully received)
    """
    if not MEDIAPIPE_AVAILABLE:
        return jsonify({
            'error': 'Movement analysis not available',
            'landmarks': [],
            'score': 0,
            'feedback': 'Camera analysis requires MediaPipe installation',
            'frame_received': False
        }), 503
    
    try:
        # Get frame data - accept both 'frame' and 'frame_data'
        frame_data = None
        exercise_type = 'general'
        json_data = request.get_json(force=True, silent=True)
        
        if 'frame' in request.files:
            file = request.files['frame']
            img_bytes = file.read()
            exercise_type = request.form.get('exercise_type', 'general')
        elif json_data:
            # Try both 'frame_data' and 'frame' keys
            frame_data = json_data.get('frame_data') or json_data.get('frame')
            if not frame_data:
                logger.error(f"No frame data in request. Keys: {json_data.keys()}")
                return jsonify({
                    'error': 'No frame data provided',
                    'frame_received': False
                }), 400
            
            # Remove data: prefix if present
            if ',' in frame_data:
                frame_data = frame_data.split(',')[1]
            
            try:
                img_bytes = base64.b64decode(frame_data)
            except Exception as decode_err:
                logger.error(f"Base64 decode error: {decode_err}")
                return jsonify({
                    'error': 'Invalid base64 encoding',
                    'frame_received': False
                }), 400
            
            # Get exercise type from json_data
            exercise_type = json_data.get('exercise_type', 'general')
        else:
            logger.error("No frame data provided in request")
            return jsonify({
                'error': 'No frame data provided',
                'frame_received': False
            }), 400
        
        # Convert to OpenCV image
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("Could not decode image from bytes")
            return jsonify({
                'error': 'Invalid image data',
                'frame_received': False
            }), 400
        
        logger.info(f"Frame received: {frame.shape}, exercise_type: {exercise_type}")
        
        # Analyze based on exercise type
        result = analyze_movement(frame, exercise_type)
        result['frame_received'] = True  # Add debug flag
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Frame analysis error: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'frame_received': False
        }), 500

def analyze_movement(frame: np.ndarray, exercise_type: str) -> Dict[str, Any]:
    """
    Analyze movement in a frame using MediaPipe.
    
    Args:
        frame: OpenCV image (BGR)
        exercise_type: Type of exercise
    
    Returns:
        Analysis results with landmarks and feedback
    """
    result = {
        'landmarks': [],
        'hand_landmarks': [],
        'score': 0,
        'feedback': '',
        'pose_detected': False,
        'hands_detected': False,
        'frame_received': True
    }
    
    try:
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        logger.debug(f"Frame converted to RGB, shape: {rgb_frame.shape}")
    except Exception as e:
        logger.error(f"Failed to convert frame to RGB: {e}")
        result['feedback'] = 'Error processing image'
        return result
    
    # Pose detection for body exercises
    if exercise_type in ['body', 'balance', 'coordination', 'general']:
        try:
            with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=0.5
            ) as pose:
                pose_results = pose.process(rgb_frame)
                
                if pose_results.pose_landmarks:
                    result['pose_detected'] = True
                    landmarks = []
                    
                    for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                        landmarks.append({
                            'id': idx,
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z,
                            'visibility': landmark.visibility
                        })
                    
                    result['landmarks'] = landmarks
                    result['score'], result['feedback'] = evaluate_pose(landmarks, exercise_type)
        except Exception as e:
            logger.error(f"Pose detection error: {e}")
    
    # Hand detection for fine motor exercises
    if exercise_type in ['hands', 'fingers', 'fine_motor', 'general']:
        try:
            with mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5
            ) as hands:
                hand_results = hands.process(rgb_frame)
                
                if hand_results.multi_hand_landmarks:
                    result['hands_detected'] = True
                    hand_landmarks = []
                    
                    for hand_idx, hand in enumerate(hand_results.multi_hand_landmarks):
                        hand_data = []
                        for idx, landmark in enumerate(hand.landmark):
                            hand_data.append({
                                'id': idx,
                                'x': landmark.x,
                                'y': landmark.y,
                                'z': landmark.z
                            })
                        hand_landmarks.append(hand_data)
                    
                    result['hand_landmarks'] = hand_landmarks
                    
                    if exercise_type != 'general':
                        hand_score, hand_feedback = evaluate_hands(hand_landmarks, exercise_type)
                        result['score'] = max(result['score'], hand_score)
                        if result['feedback']:
                            result['feedback'] += ' | ' + hand_feedback
                        else:
                            result['feedback'] = hand_feedback
        except Exception as e:
            logger.error(f"Hand detection error: {e}")
    
    if not result['pose_detected'] and not result['hands_detected']:
        result['feedback'] = 'No pose or hands detected. Please ensure you are visible in the camera.'
    
    return result

def evaluate_pose(landmarks: List[Dict], exercise_type: str) -> tuple:
    """
    Evaluate pose based on exercise type.
    
    Returns:
        (score, feedback) tuple
    """
    if not landmarks:
        return 0, "No pose detected"
    
    # Get key landmarks
    nose = landmarks[0] if len(landmarks) > 0 else None
    left_shoulder = landmarks[11] if len(landmarks) > 11 else None
    right_shoulder = landmarks[12] if len(landmarks) > 12 else None
    left_hip = landmarks[23] if len(landmarks) > 23 else None
    right_hip = landmarks[24] if len(landmarks) > 24 else None
    left_knee = landmarks[25] if len(landmarks) > 25 else None
    right_knee = landmarks[26] if len(landmarks) > 26 else None
    left_ankle = landmarks[27] if len(landmarks) > 27 else None
    right_ankle = landmarks[28] if len(landmarks) > 28 else None
    
    score = 50  # Base score
    feedback_parts = []
    
    # Check shoulder alignment
    if left_shoulder and right_shoulder:
        shoulder_diff = abs(left_shoulder['y'] - right_shoulder['y'])
        if shoulder_diff < 0.05:
            score += 15
            feedback_parts.append("Good shoulder alignment!")
        elif shoulder_diff < 0.1:
            score += 5
            feedback_parts.append("Try to level your shoulders")
        else:
            feedback_parts.append("Keep your shoulders level")
    
    # Check hip alignment  
    if left_hip and right_hip:
        hip_diff = abs(left_hip['y'] - right_hip['y'])
        if hip_diff < 0.05:
            score += 15
            feedback_parts.append("Hips are well aligned")
        elif hip_diff < 0.1:
            score += 5
        else:
            feedback_parts.append("Try to keep your hips level")
    
    # Check visibility (are they in frame properly)
    visible_count = sum(1 for lm in landmarks if lm.get('visibility', 0) > 0.5)
    visibility_ratio = visible_count / len(landmarks)
    
    if visibility_ratio > 0.8:
        score += 20
        feedback_parts.append("Great positioning!")
    elif visibility_ratio > 0.5:
        score += 10
        feedback_parts.append("Try to stay fully in frame")
    else:
        feedback_parts.append("Move back so your full body is visible")
    
    # Exercise-specific checks
    if exercise_type == 'balance':
        # Check if standing on one foot
        if left_ankle and right_ankle:
            ankle_y_diff = abs(left_ankle['y'] - right_ankle['y'])
            if ankle_y_diff > 0.1:
                score += 10
                feedback_parts.append("Good balance position!")
    
    return min(score, 100), ' | '.join(feedback_parts[:3])

def evaluate_hands(hand_landmarks: List[List[Dict]], exercise_type: str) -> tuple:
    """
    Evaluate hand movements.
    
    Returns:
        (score, feedback) tuple
    """
    if not hand_landmarks:
        return 0, "No hands detected"
    
    score = 50
    feedback_parts = []
    
    for hand_idx, hand in enumerate(hand_landmarks):
        if len(hand) < 21:
            continue
        
        # Check finger spread (for finger exercises)
        wrist = hand[0]
        thumb_tip = hand[4]
        index_tip = hand[8]
        middle_tip = hand[12]
        ring_tip = hand[16]
        pinky_tip = hand[20]
        
        # Calculate finger spread
        def distance(p1, p2):
            return ((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)**0.5
        
        spread = (
            distance(thumb_tip, index_tip) +
            distance(index_tip, middle_tip) +
            distance(middle_tip, ring_tip) +
            distance(ring_tip, pinky_tip)
        ) / 4
        
        if spread > 0.1:
            score += 20
            feedback_parts.append("Good finger spread!")
        elif spread > 0.05:
            score += 10
            feedback_parts.append("Try spreading fingers wider")
        else:
            feedback_parts.append("Spread your fingers apart")
    
    num_hands = len(hand_landmarks)
    if num_hands == 2:
        score += 10
        feedback_parts.append("Both hands detected")
    
    return min(score, 100), ' | '.join(feedback_parts[:3])

@app.route('/api/dyspraxia/session', methods=['POST'])
@require_auth
def save_dyspraxia_session():
    """
    Save a dyspraxia exercise session.
    
    Request:
        - exercise_type: Type of exercise
        - duration: Session duration in seconds
        - scores: Array of scores throughout session
        - completed: Whether exercise was completed
    """
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        session_data = {
            'user_id': request.user.get('id') or request.user.id,
            'exercise_type': data.get('exercise_type', 'general'),
            'duration': data.get('duration', 0),
            'average_score': np.mean(data.get('scores', [0])) if data.get('scores') else 0,
            'max_score': max(data.get('scores', [0])) if data.get('scores') else 0,
            'completed': data.get('completed', False),
            'created_at': datetime.utcnow().isoformat()
        }
        
        if supabase:
            result = supabase.table('dyspraxia_sessions').insert(session_data).execute()
            return jsonify({
                'success': True,
                'session_id': result.data[0]['id'] if result.data else None
            })
        else:
            # Development mode - just return success
            return jsonify({'success': True, 'session_id': 'dev-session'})
    
    except Exception as e:
        logger.error(f"Save session error: {e}")
        return jsonify({'error': str(e)}), 500

# =============================================================================
# Text Processing Endpoints
# =============================================================================

@app.route('/api/text/simplify', methods=['POST'])
@optional_auth
def simplify_text():
    """
    Simplify text for easier reading (dyslexia support).
    
    Request:
        - text: Text to simplify
        - level: Simplification level (1-3)
    
    Response:
        - simplified: Simplified text
        - reading_level: Estimated reading level
    """
    data = request.json
    
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    level = data.get('level', 2)
    
    # Basic simplification without AI
    simplified = basic_text_simplification(text, level)
    
    # Try AI simplification if available
    simplifier = get_text_simplifier()
    if simplifier and len(text) > 50:
        try:
            ai_result = simplifier(f"simplify: {text}", max_length=len(text), do_sample=False)
            if ai_result and ai_result[0].get('generated_text'):
                simplified = ai_result[0]['generated_text']
        except Exception as e:
            logger.warning(f"AI simplification failed: {e}")
    
    # Calculate reading metrics
    reading_level = calculate_reading_level(simplified)
    
    return jsonify({
        'original': text,
        'simplified': simplified,
        'reading_level': reading_level,
        'word_count': len(simplified.split())
    })

def basic_text_simplification(text: str, level: int = 2) -> str:
    """
    Basic rule-based text simplification.
    
    Args:
        text: Input text
        level: Simplification level (1=light, 2=medium, 3=heavy)
    
    Returns:
        Simplified text
    """
    # Split into sentences
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    simplified_sentences = []
    
    # Common word replacements
    simple_words = {
        'utilize': 'use',
        'approximately': 'about',
        'sufficient': 'enough',
        'commence': 'start',
        'terminate': 'end',
        'obtain': 'get',
        'purchase': 'buy',
        'demonstrate': 'show',
        'assistance': 'help',
        'difficulty': 'trouble',
        'additional': 'more',
        'numerous': 'many',
        'regarding': 'about',
        'subsequently': 'then',
        'furthermore': 'also',
        'nevertheless': 'but',
        'consequently': 'so',
        'immediately': 'now',
        'approximately': 'about',
        'frequently': 'often',
        'occasionally': 'sometimes'
    }
    
    for sentence in sentences:
        words = sentence.split()
        
        # Replace complex words
        simplified_words = []
        for word in words:
            lower_word = word.lower().strip('.,!?')
            if lower_word in simple_words:
                replacement = simple_words[lower_word]
                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                simplified_words.append(replacement + word[len(lower_word):])
            else:
                simplified_words.append(word)
        
        simplified_sentence = ' '.join(simplified_words)
        
        # Level 2+: Break long sentences
        if level >= 2 and len(simplified_words) > 15:
            # Try to split at conjunctions
            for conj in [', and ', ', but ', ', however ', '; ']:
                if conj in simplified_sentence:
                    parts = simplified_sentence.split(conj)
                    simplified_sentence = '. '.join(p.strip().capitalize() for p in parts if p.strip())
                    break
        
        # Level 3: Further simplification
        if level >= 3:
            # Remove parenthetical phrases
            simplified_sentence = re.sub(r'\([^)]*\)', '', simplified_sentence)
            # Remove some adverbs
            adverbs_to_remove = ['actually', 'basically', 'certainly', 'definitely', 'probably']
            for adv in adverbs_to_remove:
                simplified_sentence = re.sub(rf'\b{adv}\b\s*', '', simplified_sentence, flags=re.IGNORECASE)
        
        simplified_sentences.append(simplified_sentence.strip())
    
    return ' '.join(simplified_sentences)

def calculate_reading_level(text: str) -> Dict[str, Any]:
    """
    Calculate reading level metrics.
    
    Returns:
        Dictionary with reading level metrics
    """
    import re
    
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    
    if not words or not sentences:
        return {'grade_level': 0, 'difficulty': 'unknown'}
    
    # Count syllables (simple approximation)
    def count_syllables(word):
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        if count == 0:
            count = 1
        return count
    
    total_syllables = sum(count_syllables(w) for w in words)
    
    # Flesch-Kincaid Grade Level (simplified)
    avg_sentence_length = len(words) / len(sentences)
    avg_syllables_per_word = total_syllables / len(words)
    
    grade_level = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59
    grade_level = max(1, min(12, round(grade_level, 1)))
    
    # Difficulty classification
    if grade_level <= 3:
        difficulty = 'easy'
    elif grade_level <= 6:
        difficulty = 'moderate'
    elif grade_level <= 9:
        difficulty = 'challenging'
    else:
        difficulty = 'advanced'
    
    return {
        'grade_level': grade_level,
        'difficulty': difficulty,
        'avg_sentence_length': round(avg_sentence_length, 1),
        'avg_syllables_per_word': round(avg_syllables_per_word, 2)
    }

@app.route('/api/text/summarize', methods=['POST'])
@optional_auth
def summarize_text():
    """
    Summarize text.
    
    Request:
        - text: Text to summarize
        - max_length: Maximum summary length (words)
    """
    data = request.json
    
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    max_length = data.get('max_length', 100)
    
    # Try AI summarization
    summarizer = get_summarizer()
    if summarizer and len(text.split()) > 50:
        try:
            result = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
            if result and result[0].get('summary_text'):
                return jsonify({
                    'summary': result[0]['summary_text'],
                    'original_length': len(text.split()),
                    'summary_length': len(result[0]['summary_text'].split())
                })
        except Exception as e:
            logger.warning(f"AI summarization failed: {e}")
    
    # Fallback: extractive summarization
    sentences = text.split('.')
    summary_sentences = sentences[:min(3, len(sentences))]
    summary = '. '.join(s.strip() for s in summary_sentences if s.strip()) + '.'
    
    return jsonify({
        'summary': summary,
        'original_length': len(text.split()),
        'summary_length': len(summary.split())
    })

# =============================================================================
# Handwriting Analysis Endpoints
# =============================================================================

@app.route('/api/handwriting/analyze', methods=['POST'])
@optional_auth
def analyze_handwriting():
    """
    Analyze handwriting from an image.
    
    Request:
        - image: Base64 encoded image or file upload
        - expected_text: Optional expected text for comparison
    
    Response:
        - metrics: Handwriting quality metrics
        - feedback: Improvement suggestions
    """
    try:
        # Get image data
        if 'image' in request.files:
            file = request.files['image']
            img_bytes = file.read()
        elif request.json and 'image' in request.json:
            image_data = request.json['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            img_bytes = base64.b64decode(image_data)
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Convert to OpenCV image
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        expected_text = None
        if request.json:
            expected_text = request.json.get('expected_text')
        
        # Analyze handwriting
        metrics = analyze_handwriting_image(img, expected_text)
        
        return jsonify(metrics)
    
    except Exception as e:
        logger.error(f"Handwriting analysis error: {e}")
        return jsonify({'error': str(e)}), 500

def analyze_handwriting_image(img: np.ndarray, expected_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze handwriting quality from an image.
    
    Args:
        img: OpenCV image
        expected_text: Optional expected text for comparison
    
    Returns:
        Analysis metrics and feedback
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to isolate writing
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours (individual strokes/characters)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {
            'success': False,
            'error': 'No writing detected in image',
            'metrics': {},
            'feedback': ['Make sure the writing is clearly visible on the page']
        }
    
    # Calculate metrics
    heights = []
    widths = []
    positions_y = []
    angles = []
    
    for contour in contours:
        if cv2.contourArea(contour) < 50:  # Filter noise
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        heights.append(h)
        widths.append(w)
        positions_y.append(y)
        
        # Calculate slant angle
        if len(contour) > 5:
            ellipse = cv2.fitEllipse(contour)
            angles.append(ellipse[2])
    
    if not heights:
        return {
            'success': False,
            'error': 'No significant writing detected',
            'metrics': {},
            'feedback': ['Write larger characters for better analysis']
        }
    
    # Calculate consistency metrics
    height_consistency = 100 - min(100, np.std(heights) / np.mean(heights) * 100) if heights else 0
    spacing_consistency = 100 - min(100, np.std(widths) / np.mean(widths) * 100) if widths else 0
    baseline_consistency = 100 - min(100, np.std(positions_y) / np.mean(positions_y) * 100) if positions_y else 0
    slant_consistency = 100 - min(100, np.std(angles) / 45 * 100) if angles else 0
    
    # Overall score
    overall_score = (
        height_consistency * 0.25 +
        spacing_consistency * 0.25 +
        baseline_consistency * 0.30 +
        slant_consistency * 0.20
    )
    
    # Generate feedback
    feedback = []
    
    if height_consistency < 60:
        feedback.append("Try to keep your letters the same height")
    elif height_consistency > 80:
        feedback.append("Great letter height consistency!")
    
    if spacing_consistency < 60:
        feedback.append("Work on keeping even spacing between letters")
    elif spacing_consistency > 80:
        feedback.append("Good spacing between characters!")
    
    if baseline_consistency < 60:
        feedback.append("Practice writing along a straight baseline")
    elif baseline_consistency > 80:
        feedback.append("Excellent baseline alignment!")
    
    if slant_consistency < 60:
        feedback.append("Try to maintain a consistent slant in your writing")
    elif slant_consistency > 80:
        feedback.append("Nice consistent slant!")
    
    if overall_score >= 80:
        feedback.insert(0, "Excellent handwriting! Keep up the great work!")
    elif overall_score >= 60:
        feedback.insert(0, "Good progress! Keep practicing.")
    else:
        feedback.insert(0, "Keep practicing! You're improving.")
    
    return {
        'success': True,
        'metrics': {
            'overall_score': round(overall_score, 1),
            'height_consistency': round(height_consistency, 1),
            'spacing_consistency': round(spacing_consistency, 1),
            'baseline_consistency': round(baseline_consistency, 1),
            'slant_consistency': round(slant_consistency, 1),
            'character_count': len(heights)
        },
        'feedback': feedback[:5]
    }

# =============================================================================
# Email Notification Endpoints
# =============================================================================

@app.route('/api/email/send', methods=['POST'])
@require_auth
def send_email_notification():
    """
    Send email notification.
    
    Request:
        - to: Recipient email
        - subject: Email subject
        - body: Email body (HTML or plain text)
        - template: Optional template name
    """
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    required_fields = ['to', 'subject', 'body']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    try:
        result = send_email(
            to=data['to'],
            subject=data['subject'],
            body=data['body'],
            is_html=data.get('is_html', True)
        )
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Email send error: {e}")
        return jsonify({'error': str(e)}), 500

def send_email(to: str, subject: str, body: str, is_html: bool = True) -> Dict[str, Any]:
    """
    Send an email using SMTP.
    
    Args:
        to: Recipient email address
        subject: Email subject
        body: Email body
        is_html: Whether body is HTML
    
    Returns:
        Result dictionary
    """
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    if not Config.SMTP_USER or not Config.SMTP_PASSWORD:
        logger.warning("SMTP not configured - email not sent")
        return {'success': False, 'error': 'Email service not configured'}
    
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = Config.EMAIL_FROM
        msg['To'] = to
        msg['Subject'] = subject
        
        content_type = 'html' if is_html else 'plain'
        msg.attach(MIMEText(body, content_type))
        
        with smtplib.SMTP(Config.SMTP_HOST, Config.SMTP_PORT) as server:
            server.starttls()
            server.login(Config.SMTP_USER, Config.SMTP_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"Email sent to {to}")
        return {'success': True, 'message': 'Email sent successfully'}
    
    except Exception as e:
        logger.error(f"SMTP error: {e}")
        return {'success': False, 'error': str(e)}

# =============================================================================
# Report Generation Endpoints
# =============================================================================

@app.route('/api/reports/progress', methods=['GET'])
@require_auth
def generate_progress_report():
    """
    Generate a progress report PDF.
    
    Query params:
        - user_id: User ID (optional, defaults to current user)
        - start_date: Report start date
        - end_date: Report end date
        - module: Specific module or 'all'
    """
    user_id = request.args.get('user_id') or request.user.get('id') or request.user.id
    start_date = request.args.get('start_date', (datetime.utcnow() - timedelta(days=30)).isoformat())
    end_date = request.args.get('end_date', datetime.utcnow().isoformat())
    module = request.args.get('module', 'all')
    
    try:
        # Fetch user data
        user_data = {}
        progress_data = {}
        
        if supabase:
            # Get user profile
            profile_result = supabase.table('profiles').select('*').eq('id', user_id).single().execute()
            if profile_result.data:
                user_data = profile_result.data
            
            # Get progress data
            progress_query = supabase.table('progress').select('*').eq('user_id', user_id)
            progress_query = progress_query.gte('created_at', start_date).lte('created_at', end_date)
            
            if module != 'all':
                progress_query = progress_query.eq('module', module)
            
            progress_result = progress_query.execute()
            progress_data = progress_result.data or []
        
        # Generate report data
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'user': {
                'name': user_data.get('full_name', 'Student'),
                'id': user_id
            },
            'period': {
                'start': start_date,
                'end': end_date
            },
            'summary': generate_report_summary(progress_data),
            'modules': generate_module_breakdown(progress_data),
            'recommendations': generate_recommendations(progress_data)
        }
        
        return jsonify(report)
    
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return jsonify({'error': str(e)}), 500

def generate_report_summary(progress_data: List[Dict]) -> Dict[str, Any]:
    """Generate summary statistics from progress data."""
    if not progress_data:
        return {
            'total_sessions': 0,
            'total_time_minutes': 0,
            'average_score': 0,
            'improvement': 0
        }
    
    total_sessions = len(progress_data)
    total_time = sum(p.get('duration', 0) for p in progress_data) / 60  # Convert to minutes
    scores = [p.get('score', 0) for p in progress_data]
    avg_score = np.mean(scores) if scores else 0
    
    # Calculate improvement (compare first half to second half)
    mid = len(scores) // 2
    if mid > 0:
        first_half_avg = np.mean(scores[:mid])
        second_half_avg = np.mean(scores[mid:])
        improvement = ((second_half_avg - first_half_avg) / first_half_avg * 100) if first_half_avg > 0 else 0
    else:
        improvement = 0
    
    return {
        'total_sessions': total_sessions,
        'total_time_minutes': round(total_time, 1),
        'average_score': round(avg_score, 1),
        'improvement': round(improvement, 1)
    }

def generate_module_breakdown(progress_data: List[Dict]) -> Dict[str, Dict]:
    """Generate per-module statistics."""
    modules = {}
    
    for entry in progress_data:
        module = entry.get('module', 'unknown')
        if module not in modules:
            modules[module] = {
                'sessions': 0,
                'total_time': 0,
                'scores': [],
                'activities': {}
            }
        
        modules[module]['sessions'] += 1
        modules[module]['total_time'] += entry.get('duration', 0)
        modules[module]['scores'].append(entry.get('score', 0))
        
        activity = entry.get('activity', 'general')
        if activity not in modules[module]['activities']:
            modules[module]['activities'][activity] = 0
        modules[module]['activities'][activity] += 1
    
    # Calculate averages
    for module in modules:
        scores = modules[module]['scores']
        modules[module]['average_score'] = round(np.mean(scores), 1) if scores else 0
        modules[module]['total_time_minutes'] = round(modules[module]['total_time'] / 60, 1)
        del modules[module]['scores']
    
    return modules

def generate_recommendations(progress_data: List[Dict]) -> List[str]:
    """Generate personalized recommendations."""
    recommendations = []
    
    if not progress_data:
        recommendations.append("Start with short, daily practice sessions to build consistency")
        recommendations.append("Try different modules to find what works best")
        return recommendations
    
    # Analyze patterns
    modules = {}
    for entry in progress_data:
        module = entry.get('module', 'unknown')
        if module not in modules:
            modules[module] = []
        modules[module].append(entry.get('score', 0))
    
    # Module-specific recommendations
    for module, scores in modules.items():
        avg_score = np.mean(scores)
        
        if avg_score < 50:
            recommendations.append(f"Focus more time on {module} exercises - practice makes perfect!")
        elif avg_score < 70:
            recommendations.append(f"Good progress in {module}! Try more challenging activities.")
        else:
            recommendations.append(f"Excellent work in {module}! Consider helping others or trying advanced exercises.")
    
    # General recommendations
    total_sessions = len(progress_data)
    if total_sessions < 10:
        recommendations.append("Try to practice more regularly - aim for daily sessions")
    
    if len(modules) < 2:
        recommendations.append("Explore other modules for well-rounded development")
    
    return recommendations[:5]

# =============================================================================
# Activity Logging
# =============================================================================

@app.route('/api/activity/log', methods=['POST'])
@require_auth
def log_activity():
    """
    Log a user activity.
    
    Request:
        - activity_type: Type of activity
        - module: Module name
        - details: Activity details
        - score: Optional score
        - duration: Optional duration in seconds
    """
    data = request.json
    
    if not data or 'activity_type' not in data:
        return jsonify({'error': 'Activity type required'}), 400
    
    try:
        activity_data = {
            'user_id': request.user.get('id') or request.user.id,
            'activity_type': data['activity_type'],
            'module': data.get('module'),
            'details': data.get('details', {}),
            'score': data.get('score'),
            'duration': data.get('duration'),
            'created_at': datetime.utcnow().isoformat()
        }
        
        if supabase:
            result = supabase.table('activity_logs').insert(activity_data).execute()
            return jsonify({
                'success': True,
                'activity_id': result.data[0]['id'] if result.data else None
            })
        else:
            return jsonify({'success': True, 'activity_id': 'dev-activity'})
    
    except Exception as e:
        logger.error(f"Activity log error: {e}")
        return jsonify({'error': str(e)}), 500

# =============================================================================
# File Upload Endpoints
# =============================================================================

@app.route('/api/upload/avatar', methods=['POST'])
@require_auth
def upload_avatar():
    """
    Upload user avatar image.
    
    Request:
        - file: Image file (multipart/form-data)
    
    Response:
        - url: Public URL of uploaded avatar
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Validate file type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    
    if ext not in allowed_extensions:
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        user_id = request.user.get('id') or request.user.id
        filename = f"avatars/{user_id}.{ext}"
        
        if supabase:
            # Upload to Supabase Storage
            file_data = file.read()
            result = supabase.storage.from_('avatars').upload(
                filename,
                file_data,
                {'content-type': file.content_type, 'upsert': 'true'}
            )
            
            # Get public URL
            public_url = supabase.storage.from_('avatars').get_public_url(filename)
            
            # Update profile
            supabase.table('profiles').update({
                'avatar_url': public_url
            }).eq('id', user_id).execute()
            
            return jsonify({
                'success': True,
                'url': public_url
            })
        else:
            return jsonify({
                'success': True,
                'url': f'/uploads/{filename}'
            })
    
    except Exception as e:
        logger.error(f"Avatar upload error: {e}")
        return jsonify({'error': str(e)}), 500

# =============================================================================
# Patient-Doctor Auto Assignment Logic
# =============================================================================

# Default doctor UUID (replace with your actual default doctor id)
DEFAULT_DOCTOR_ID = os.getenv('DEFAULT_DOCTOR_ID', '00000000-0000-0000-0000-000000000000')

def assign_patient_to_doctor(patient_id):
    if not supabase:
        logger.warning('Supabase client not available for patient assignment')
        return
    # Check if patient is already assigned
    result = supabase.table('doctor_patients').select('id').eq('patient_id', patient_id).execute()
    if result.data and len(result.data) > 0:
        return  # Already assigned

    # Determine doctor_id: prefer DEFAULT_DOCTOR_ID if set to a real id, otherwise pick first available doctor
    doctor_id = DEFAULT_DOCTOR_ID

    # If DEFAULT_DOCTOR_ID is unset or placeholder, try to pick an available doctor
    if not doctor_id or doctor_id == '00000000-0000-0000-0000-000000000000':
        try:
            docs = supabase.table('profiles').select('id').eq('role', 'doctor').eq('is_available', True).limit(1).execute()
            if docs.data and len(docs.data) > 0:
                doctor_id = docs.data[0]['id']
        except Exception as e:
            logger.warning(f"Failed to pick fallback doctor: {e}")

    if not doctor_id or doctor_id == '00000000-0000-0000-0000-000000000000':
        logger.warning('No valid doctor id found to assign patient')
        return

    # Insert assignment
    supabase.table('doctor_patients').insert({
        'doctor_id': doctor_id,
        'patient_id': patient_id,
        'relationship_type': 'assigned',
        'status': 'active'
    }).execute()

@app.route('/api/patient/login', methods=['POST'])
def patient_login():
    """
    Patient login endpoint. After successful login, auto-assign patient to doctor if not already assigned.
    Request: { email, password }
    """
    data = request.json
    email = data.get('email')
    password = data.get('password')
    if not email or not password:
        return jsonify({'error': 'Missing email or password'}), 400
    try:
        # Authenticate patient (replace with your actual auth logic)
        # NOTE: role in DB uses 'student' for patients
        user = supabase.table('profiles').select('*').eq('email', email).eq('role', 'student').execute()
        if not user.data or len(user.data) == 0:
            return jsonify({'error': 'Patient not found'}), 404
        patient = user.data[0]
        # TODO: Add password check if needed
        assign_patient_to_doctor(patient['id'])
        return jsonify({'success': True, 'patient': patient})
    except Exception as e:
        logger.error(f"Patient login error: {e}")
        return jsonify({'error': str(e)}), 500

# =============================================================================
# Bulk Assign Unassigned Patients to Default Doctor
# =============================================================================

@app.route('/api/doctor/assign-all-patients', methods=['POST'])
def assign_all_patients():
    """
    Assign all unassigned patients to the default doctor.
    Returns a list of assigned patients.
    """
    if not supabase:
        return jsonify({'error': 'Supabase client not available'}), 500
    try:
        # Get all patient/student profiles
        patients = supabase.table('profiles').select('id').eq('role', 'student').execute()
        assigned = []
        for patient in patients.data:
            pid = patient['id']
            # Check if already assigned
            result = supabase.table('doctor_patients').select('id').eq('patient_id', pid).execute()
            if not result.data or len(result.data) == 0:
                # Assign to default doctor (use fallback logic in assign_patient_to_doctor)
                assign_patient_to_doctor(pid)
                assigned.append(pid)
        return jsonify({'success': True, 'assigned_patients': assigned})
    except Exception as e:
        logger.error(f"Bulk assign error: {e}")
        return jsonify({'error': str(e)}), 500

# =============================================================================
# Script to Assign All Unassigned Patients/Students to Dr. Anamika Khandelwal
# =============================================================================

@app.route('/api/doctor/assign-all-to-anamika', methods=['POST'])
def assign_all_to_anamika():
    """
    Assign all unassigned patients and students to Dr. Anamika Khandelwal.
    Returns a list of assigned patient IDs.
    """
    if not supabase:
        return jsonify({'error': 'Supabase client not available'}), 500
    try:
        # Find Dr. Anamika Khandelwal's profile
        doctor = supabase.table('profiles').select('id').eq('full_name', 'Anamika Khandelwal').eq('role', 'doctor').execute()
        if not doctor.data or len(doctor.data) == 0:
            return jsonify({'error': 'Dr. Anamika Khandelwal not found'}), 404
        doctor_id = doctor.data[0]['id']
        # Get all patient and student profiles
        patients = supabase.table('profiles').select('id').in_('role', ['patient', 'student']).execute()
        assigned = []
        for patient in patients.data:
            pid = patient['id']
            # Check if already assigned
            result = supabase.table('doctor_patients').select('id').eq('patient_id', pid).eq('doctor_id', doctor_id).execute()
            if not result.data or len(result.data) == 0:
                # Assign to Dr. Anamika Khandelwal
                supabase.table('doctor_patients').insert({
                    'doctor_id': doctor_id,
                    'patient_id': pid,
                    'relationship_type': 'assigned',
                    'status': 'active'
                }).execute()
                assigned.append(pid)
        return jsonify({'success': True, 'assigned_patients': assigned})
    except Exception as e:
        logger.error(f"Bulk assign to Anamika error: {e}")
        return jsonify({'error': str(e)}), 500

# =============================================================================
# Script to List All Patients and Their Doctor Assignments
# =============================================================================

@app.route('/api/doctor/list-patient-assignments', methods=['GET'])
def list_patient_assignments():
    """
    List all patients and their doctor assignments.
    Returns a list of patients with doctor_id (or None if unassigned).
    """
    if not supabase:
        return jsonify({'error': 'Supabase client not available'}), 500
    try:
        patients = supabase.table('profiles').select('id, full_name, email').eq('role', 'student').execute()
        assignments = supabase.table('doctor_patients').select('doctor_id, patient_id').execute()
        patient_map = {p['id']: p for p in patients.data}
        assignment_map = {a['patient_id']: a['doctor_id'] for a in assignments.data}
        result = []
        for pid, patient in patient_map.items():
            doctor_id = assignment_map.get(pid)
            result.append({
                'patient_id': pid,
                'full_name': patient.get('full_name'),
                'email': patient.get('email'),
                'doctor_id': doctor_id
            })
        return jsonify({'patients': result})
    except Exception as e:
        logger.error(f"List assignments error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/doctors', methods=['GET'])
def list_doctors():
    """Return list of active doctors (service endpoint).

    Frontend can use this to show available doctors to patients.
    """
    if not supabase:
        return jsonify({'error': 'Supabase client not available'}), 500
    try:
        docs = supabase.table('profiles').select('id, full_name, display_name, avatar_url, specialization, is_available').eq('role', 'doctor').execute()
        return jsonify({'doctors': docs.data})
    except Exception as e:
        logger.error(f"List doctors error: {e}")
        return jsonify({'error': str(e)}), 500

# =============================================================================
# Error Handlers
# =============================================================================

@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': 'Bad request'}), 400

@app.errorhandler(401)
def unauthorized(e):
    return jsonify({'error': 'Unauthorized'}), 401

@app.errorhandler(403)
def forbidden(e):
    return jsonify({'error': 'Forbidden'}), 403

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting Cogno Solution API on port {port}")
    logger.info(f"MediaPipe available: {MEDIAPIPE_AVAILABLE}")
    logger.info(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
    logger.info(f"Supabase configured: {supabase is not None}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
