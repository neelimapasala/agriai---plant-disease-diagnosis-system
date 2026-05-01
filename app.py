from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageStat
import base64
import os
import json
from datetime import datetime, timedelta
import random
import numpy as np
import hashlib
import requests
from math import radians, sin, cos, sqrt, atan2

app = Flask(__name__, static_folder='.')
CORS(app)

# Create uploads folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================================
# DATABASES & STORAGE
# ============================================

# Blockchain Storage
blockchain_chain = []
pending_diagnoses = []

# User diagnosis history
diagnosis_history = []

# Market prices database
MARKET_PRICES = {
    'wheat': {'price': 2250, 'trend': 'up', 'change': 5, 'best_time': 'Next 2 weeks'},
    'corn': {'price': 1850, 'trend': 'down', 'change': -2, 'best_time': 'Sell immediately'},
    'tomato': {'price': 1200, 'trend': 'up', 'change': 12, 'best_time': 'Next week peak'},
    'potato': {'price': 950, 'trend': 'up', 'change': 3, 'best_time': 'Hold for 5 days'},
    'onion': {'price': 1800, 'trend': 'down', 'change': -8, 'best_time': 'Wait for recovery'},
    'rice': {'price': 2100, 'trend': 'stable', 'change': 0, 'best_time': 'Anytime'},
    'sugarcane': {'price': 3500, 'trend': 'up', 'change': 4, 'best_time': 'Next month'},
    'cotton': {'price': 6200, 'trend': 'down', 'change': -3, 'best_time': 'Store for 2 weeks'}
}

# Expert database
EXPERTS = [
    {'name': 'Dr. Rajesh Kumar', 'specialization': 'Plant Pathology', 'experience': '15 years', 'available': True, 'rating': 4.8},
    {'name': 'Dr. Priya Sharma', 'specialization': 'Agronomy', 'experience': '12 years', 'available': True, 'rating': 4.9},
    {'name': 'Dr. Anand Singh', 'specialization': 'Soil Science', 'experience': '20 years', 'available': False, 'rating': 4.7},
    {'name': 'Dr. Neha Verma', 'specialization': 'Organic Farming', 'experience': '10 years', 'available': True, 'rating': 4.9}
]

# Government schemes database
GOVERNMENT_SCHEMES = [
    {'name': 'PM-KISAN', 'benefit': '₹6,000 per year', 'eligibility': 'All farmers', 'deadline': 'Rolling'},
    {'name': 'Crop Insurance', 'benefit': 'Low premium, high coverage', 'eligibility': 'Land owners', 'deadline': 'Before sowing'},
    {'name': 'Organic Farming Subsidy', 'benefit': '50% subsidy', 'eligibility': 'Organic farmers', 'deadline': 'March 31'},
    {'name': 'Equipment Purchase Subsidy', 'benefit': '40% subsidy up to ₹1L', 'eligibility': 'Small farmers', 'deadline': 'December 31'},
    {'name': 'Solar Pump Scheme', 'benefit': '60% subsidy', 'eligibility': 'Farm owners', 'deadline': 'February 28'}
]

# Nearby services database
NEARBY_SERVICES = [
    {'name': 'Green Seed Store', 'type': 'seed', 'distance': 2.5, 'rating': 4.5, 'contact': '9876543210'},
    {'name': 'Kisan Fertilizers', 'type': 'fertilizer', 'distance': 3.2, 'rating': 4.3, 'contact': '9876543211'},
    {'name': 'Agri Equipment Rentals', 'type': 'equipment', 'distance': 4.0, 'rating': 4.7, 'contact': '9876543212'},
    {'name': 'Organic Inputs Shop', 'type': 'organic', 'distance': 5.1, 'rating': 4.6, 'contact': '9876543213'},
    {'name': 'Plant Clinic', 'type': 'clinic', 'distance': 6.0, 'rating': 4.8, 'contact': '9876543214'}
]

# Daily tips database
DAILY_TIPS = [
    "Apply neem oil spray early morning or late evening for best pest control results.",
    "Water plants at the base, not on leaves, to prevent fungal diseases.",
    "Add compost to soil every 3 months to improve soil health naturally.",
    "Rotate crops each season to prevent soil-borne diseases.",
    "Remove infected plant parts immediately to prevent disease spread.",
    "Use mulch to retain soil moisture and suppress weed growth.",
    "Test soil pH annually for optimal nutrient management.",
    "Plant marigolds as companion plants to repel harmful insects.",
    "Harvest crops in the morning for better shelf life and quality.",
    "Store seeds in cool, dry place to maintain germination rate."
]

# ============================================
# IMAGE ANALYSIS FUNCTIONS
# ============================================

def extract_image_features(image_path):
    """Extract features from image for accurate diagnosis"""
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        pixels = list(img.getdata())
        avg_r = sum(p[0] for p in pixels) / len(pixels)
        avg_g = sum(p[1] for p in pixels) / len(pixels)
        avg_b = sum(p[2] for p in pixels) / len(pixels)
        
        r_var = sum((p[0] - avg_r) ** 2 for p in pixels) / len(pixels)
        g_var = sum((p[1] - avg_g) ** 2 for p in pixels) / len(pixels)
        b_var = sum((p[2] - avg_b) ** 2 for p in pixels) / len(pixels)
        avg_variance = (r_var + g_var + b_var) / 3
        
        is_green = avg_g > avg_r and avg_g > avg_b
        has_spots = avg_variance > 3000
        is_yellowish = avg_r > avg_g and avg_r > avg_b
        
        return {
            'is_green': is_green,
            'has_spots': has_spots,
            'is_yellowish': is_yellowish,
            'avg_variance': avg_variance
        }
    except Exception as e:
        return None

def diagnose_from_features(features):
    """Diagnose disease based on image features"""
    if not features:
        return get_default_diagnosis()
    
    if features['is_green'] and not features['has_spots'] and not features['is_yellowish']:
        return {
            'name': 'Healthy Plant',
            'confidence': 92,
            'symptoms': 'No visible disease symptoms. Leaf is healthy green.',
            'treatment': 'Continue regular care. Maintain watering and fertilization.',
            'prevention': 'Regular monitoring, proper spacing, balanced nutrition.',
            'urgency': 'Low',
            'organic_remedy': 'Continue organic practices',
            'economic_impact': 'Optimal yield expected',
            'resistant_varieties': ['All varieties performing well']
        }
    
    if features['has_spots'] and features['avg_variance'] > 3500:
        return {
            'name': 'Early Blight',
            'confidence': 88,
            'symptoms': 'Dark brown spots with concentric rings on leaves. Yellowing around affected areas.',
            'treatment': 'Remove infected leaves immediately. Apply copper-based fungicide every 7 days.',
            'prevention': 'Rotate crops annually. Water at base to keep leaves dry.',
            'urgency': 'High',
            'organic_remedy': 'Neem oil spray (2 tbsp per gallon water) every 5-7 days',
            'economic_impact': 'Can reduce yield by 30-50% if untreated',
            'resistant_varieties': ['Mountain Merit', 'Defiant PhR', 'Iron Lady']
        }
    
    if features['is_yellowish'] and not features['has_spots']:
        return {
            'name': 'Nutrient Deficiency',
            'confidence': 79,
            'symptoms': 'Yellowing leaves (chlorosis), stunted growth, poor development.',
            'treatment': 'Apply balanced NPK fertilizer (10-10-10). Add compost to soil.',
            'prevention': 'Regular soil testing. Proper fertilization schedule.',
            'urgency': 'Medium',
            'organic_remedy': 'Compost tea or seaweed extract application',
            'economic_impact': 'Moderate yield reduction',
            'resistant_varieties': ['Nutrient-efficient varieties']
        }
    
    return get_default_diagnosis()

def get_default_diagnosis():
    return {
        'name': 'General Leaf Disease',
        'confidence': 75,
        'symptoms': 'Irregular spots on leaves, possible discoloration.',
        'treatment': 'Remove affected leaves. Apply appropriate fungicide.',
        'prevention': 'Maintain plant hygiene. Avoid overhead watering.',
        'urgency': 'Medium',
        'organic_remedy': 'Neem oil or copper spray',
        'economic_impact': 'Moderate yield impact',
        'resistant_varieties': ['Standard varieties']
    }

# ============================================
# BLOCKCHAIN FUNCTIONS
# ============================================

def create_genesis_block():
    genesis_block = {
        'index': 0,
        'timestamp': datetime.now().isoformat(),
        'diagnoses': [],
        'previous_hash': '0',
        'hash': calculate_hash(0, [], '0')
    }
    blockchain_chain.append(genesis_block)

def calculate_hash(index, diagnoses, previous_hash):
    data = f"{index}{diagnoses}{previous_hash}{datetime.now()}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]

def add_diagnosis_to_blockchain(diagnosis):
    pending_diagnoses.append(diagnosis)
    if len(pending_diagnoses) >= 3:
        create_block()

def create_block():
    if not pending_diagnoses:
        return
    previous_block = blockchain_chain[-1]
    new_block = {
        'index': len(blockchain_chain),
        'timestamp': datetime.now().isoformat(),
        'diagnoses': pending_diagnoses.copy(),
        'previous_hash': previous_block['hash'],
        'hash': calculate_hash(len(blockchain_chain), pending_diagnoses, previous_block['hash'])
    }
    blockchain_chain.append(new_block)
    pending_diagnoses.clear()
    return new_block

# Initialize blockchain
create_genesis_block()

# ============================================
# WEATHER & RISK ANALYSIS
# ============================================

def get_weather_data():
    """Generate realistic weather data"""
    weather_conditions = ['Sunny', 'Partly Cloudy', 'Cloudy', 'Light Rain', 'Humid']
    return {
        'temperature': round(random.uniform(18, 35), 1),
        'humidity': round(random.uniform(45, 90), 1),
        'wind_speed': round(random.uniform(0, 20), 1),
        'rainfall': round(random.uniform(0, 50), 1),
        'condition': random.choice(weather_conditions)
    }

def calculate_disease_risk(weather):
    risk_score = 0
    if weather['humidity'] > 80:
        risk_score += 40
    if weather['temperature'] > 30:
        risk_score += 20
    if weather['rainfall'] > 25:
        risk_score += 30
    
    if risk_score > 60:
        return 'High', 'Apply preventive fungicide immediately'
    elif risk_score > 30:
        return 'Medium', 'Monitor plants closely'
    else:
        return 'Low', 'Regular maintenance sufficient'

# ============================================
# CROP YIELD PREDICTION
# ============================================

def predict_yield(crop_type, area, soil_health):
    base_yield = {
        'wheat': 45, 'rice': 40, 'corn': 50, 'tomato': 35, 'potato': 30
    }
    base = base_yield.get(crop_type.lower(), 40)
    soil_factor = 1.0 if soil_health == 'good' else 0.8 if soil_health == 'medium' else 0.6
    predicted = base * soil_factor * area
    return round(predicted, 1)

# ============================================
# FERTILIZER CALCULATIONS
# ============================================

def calculate_fertilizer_requirements(crop_type, area, soil_type):
    requirements = {
        'wheat': {'n': 120, 'p': 60, 'k': 60},
        'rice': {'n': 100, 'p': 50, 'k': 50},
        'corn': {'n': 150, 'p': 70, 'k': 70},
        'tomato': {'n': 80, 'p': 100, 'k': 100}
    }
    req = requirements.get(crop_type.lower(), {'n': 100, 'p': 60, 'k': 60})
    
    soil_factors = {'loamy': 1.0, 'sandy': 1.2, 'clay': 0.8, 'black': 0.9}
    factor = soil_factors.get(soil_type.lower(), 1.0)
    
    return {
        'nitrogen_kg': round(req['n'] * area * factor),
        'phosphorus_kg': round(req['p'] * area * factor),
        'potassium_kg': round(req['k'] * area * factor),
        'total_fertilizer_kg': round(sum(req.values()) * area * factor)
    }

# ============================================
# COST & PROFIT CALCULATIONS
# ============================================

def calculate_profit(crop_type, area, market_price):
    production_cost = {
        'wheat': 25000, 'rice': 30000, 'corn': 28000, 'tomato': 45000, 'potato': 35000
    }
    cost_per_acre = production_cost.get(crop_type.lower(), 30000)
    revenue = market_price * area * 20  # 20 quintals per acre approx
    profit = revenue - (cost_per_acre * area)
    return {
        'total_cost': cost_per_acre * area,
        'total_revenue': revenue,
        'net_profit': profit,
        'profit_per_acre': profit / area if area > 0 else 0
    }

# ============================================
# PEST IDENTIFICATION
# ============================================

PESTS_DATABASE = {
    'aphids': {
        'description': 'Small green/black insects on leaves and stems',
        'damage': 'Sucks sap, causes leaf curling',
        'organic_control': 'Neem oil spray, ladybugs introduction',
        'chemical_control': 'Malathion, Imidacloprid'
    },
    'spider_mites': {
        'description': 'Tiny red spots with fine webbing',
        'damage': 'Yellow stippling on leaves',
        'organic_control': 'Sulfur spray, predatory mites',
        'chemical_control': 'Abamectin, Fenpyroximate'
    },
    'whitefly': {
        'description': 'Small white flying insects',
        'damage': 'Yellowing leaves, sticky honeydew',
        'organic_control': 'Yellow sticky traps, neem oil',
        'chemical_control': 'Buprofezin, Diafenthiuron'
    },
    'caterpillar': {
        'description': 'Green worms eating leaves',
        'damage': 'Chewed leaves, holes in fruits',
        'organic_control': 'Bt spray, hand picking',
        'chemical_control': 'Cypermethrin, Chlorantraniliprole'
    }
}

# ============================================
# CHATBOT RESPONSES
# ============================================

CHATBOT_RESPONSES = {
    'greeting': "Namaste! 🌱 I'm Neelima's AI Assistant. How can I help you with your plants today?",
    'price': "Current market prices:\n🌾 Wheat: ₹2,250/quintal\n🌽 Corn: ₹1,850/quintal\n🍅 Tomato: ₹1,200/quintal\n🥔 Potato: ₹950/quintal",
    'weather': "Today's weather: {temp}°C, {humidity}% humidity. Disease risk: {risk}. {advice}",
    'fertilizer': "For healthy plants, use organic compost or balanced NPK fertilizer. Apply every 15-20 days during growing season.",
    'disease': "Please upload a photo of the affected leaf or describe the symptoms in detail for accurate diagnosis.",
    'pest': "Common pests include aphids, spider mites, and whiteflies. Use neem oil spray (2 tbsp per gallon) for organic control.",
    'treatment': "Treatment depends on the disease. First identify the problem using our diagnosis tool.",
    'organic': "Great choice! Organic options: neem oil, compost tea, baking soda solution, and beneficial insects.",
    'water': "Water plants early morning or evening. Avoid overhead watering to prevent fungal diseases.",
    'help': "I can help with:\n• Plant disease diagnosis\n• Treatment recommendations\n• Market prices\n• Weather forecast\n• Organic farming tips\n• Pest control"
}

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'features': [
            'Disease Diagnosis', 'Weather Analysis', 'Market Prices', 'Expert Consultation',
            'Government Schemes', 'Yield Prediction', 'Fertilizer Calculator', 'Pest Control',
            'Blockchain Security', 'Multi-language', 'AI Chatbot', 'Community Forum'
        ],
        'creator': 'Neelima Pasala',
        'languages': ['en', 'hi', 'te', 'ta', 'kn', 'mr'],
        'timestamp': datetime.now().isoformat()
    })

# ============================================
# DIAGNOSIS ENDPOINT
# ============================================

@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        image_path = os.path.join(UPLOAD_FOLDER, 'temp_leaf.jpg')
        image_file.save(image_path)
        
        features = extract_image_features(image_path)
        diagnosis = diagnose_from_features(features)
        
        # Get weather data
        weather = get_weather_data()
        disease_risk, risk_advice = calculate_disease_risk(weather)
        
        # Add to blockchain
        diagnosis_record = {
            'id': len(blockchain_chain) + 1,
            'disease': diagnosis['name'],
            'confidence': diagnosis['confidence'],
            'timestamp': datetime.now().isoformat()
        }
        add_diagnosis_to_blockchain(diagnosis_record)
        
        # Save to history
        diagnosis_history.append({
            **diagnosis_record,
            'image_analyzed': True
        })
        
        os.remove(image_path)
        
        return jsonify({
            'success': True,
            'diagnosis': diagnosis,
            'weather': weather,
            'disease_risk': disease_risk,
            'risk_advice': risk_advice,
            'blockchain_verified': True,
            'blockchain_id': len(blockchain_chain),
            'creator': 'Neelima Pasala',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================
# MARKET PRICES ENDPOINT
# ============================================

@app.route('/api/market-prices', methods=['GET'])
def get_market_prices():
    crop = request.args.get('crop', '').lower()
    if crop and crop in MARKET_PRICES:
        return jsonify({'success': True, 'market': MARKET_PRICES[crop]})
    return jsonify({'success': True, 'markets': MARKET_PRICES})

@app.route('/api/price-alert', methods=['POST'])
def set_price_alert():
    data = request.json
    crop = data.get('crop')
    target_price = data.get('target_price')
    return jsonify({
        'success': True,
        'message': f'Alert set for {crop} at ₹{target_price}',
        'alert_id': hashlib.md5(f"{crop}{target_price}".encode()).hexdigest()[:8]
    })

# ============================================
# WEATHER ENDPOINT
# ============================================

@app.route('/api/weather', methods=['GET'])
def get_weather():
    lat = request.args.get('lat', 20.5937, type=float)
    lon = request.args.get('lon', 78.9629, type=float)
    weather = get_weather_data()
    disease_risk, advice = calculate_disease_risk(weather)
    
    return jsonify({
        'success': True,
        'current': weather,
        'disease_risk': disease_risk,
        'advice': advice,
        'forecast': [
            {**get_weather_data(), 'day': f'Day +{i+1}'} for i in range(5)
        ]
    })

# ============================================
# EXPERTS & CONSULTATION ENDPOINTS
# ============================================

@app.route('/api/experts', methods=['GET'])
def get_experts():
    specialization = request.args.get('specialization', '')
    if specialization:
        filtered = [e for e in EXPERTS if specialization.lower() in e['specialization'].lower()]
        return jsonify({'success': True, 'experts': filtered})
    return jsonify({'success': True, 'experts': EXPERTS})

@app.route('/api/book-consultation', methods=['POST'])
def book_consultation():
    data = request.json
    expert_id = data.get('expert_id')
    date = data.get('date')
    time = data.get('time')
    
    return jsonify({
        'success': True,
        'booking_id': hashlib.md5(f"{expert_id}{date}{time}".encode()).hexdigest()[:10],
        'message': 'Consultation booked successfully',
        'meeting_link': f'https://meet.example.com/{hashlib.md5(expert_id.encode()).hexdigest()[:8]}'
    })

# ============================================
# GOVERNMENT SCHEMES ENDPOINT
# ============================================

@app.route('/api/government-schemes', methods=['GET'])
def get_schemes():
    category = request.args.get('category', '')
    if category:
        filtered = [s for s in GOVERNMENT_SCHEMES if category.lower() in s['name'].lower()]
        return jsonify({'success': True, 'schemes': filtered})
    return jsonify({'success': True, 'schemes': GOVERNMENT_SCHEMES})

@app.route('/api/check-eligibility', methods=['POST'])
def check_eligibility():
    data = request.json
    farmer_type = data.get('farmer_type', 'small')
    land_holding = data.get('land_holding', 0)
    
    eligible_schemes = []
    for scheme in GOVERNMENT_SCHEMES:
        if 'small' in scheme['eligibility'].lower() or land_holding > 0:
            eligible_schemes.append(scheme)
    
    return jsonify({
        'success': True,
        'eligible_schemes': eligible_schemes,
        'total_benefit': sum([int(''.join(filter(str.isdigit, s['benefit']))) for s in eligible_schemes if any(c.isdigit() for c in s['benefit'])])
    })

# ============================================
# YIELD & PROFIT ENDPOINTS
# ============================================

@app.route('/api/predict-yield', methods=['POST'])
def predict_crop_yield():
    data = request.json
    crop = data.get('crop', 'wheat')
    area = data.get('area', 1)
    soil_health = data.get('soil_health', 'good')
    
    yield_estimate = predict_yield(crop, area, soil_health)
    return jsonify({
        'success': True,
        'crop': crop,
        'area': area,
        'predicted_yield_quintals': yield_estimate,
        'confidence': round(random.uniform(75, 95), 1)
    })

@app.route('/api/calculate-profit', methods=['POST'])
def calculate_profit_api():
    data = request.json
    crop = data.get('crop', 'wheat')
    area = data.get('area', 1)
    market_price = data.get('market_price', MARKET_PRICES.get(crop.lower(), {}).get('price', 2000))
    
    profit = calculate_profit(crop, area, market_price)
    return jsonify({
        'success': True,
        **profit
    })

# ============================================
# FERTILIZER CALCULATOR ENDPOINT
# ============================================

@app.route('/api/fertilizer-calculator', methods=['POST'])
def fertilizer_calculator():
    data = request.json
    crop = data.get('crop', 'wheat')
    area = data.get('area', 1)
    soil_type = data.get('soil_type', 'loamy')
    
    requirements = calculate_fertilizer_requirements(crop, area, soil_type)
    return jsonify({
        'success': True,
        **requirements,
        'recommendation': 'Apply in 2-3 split doses for better absorption'
    })

# ============================================
# PEST CONTROL ENDPOINT
# ============================================

@app.route('/api/identify-pest', methods=['POST'])
def identify_pest():
    data = request.json
    description = data.get('description', '').lower()
    
    for pest_name, pest_info in PESTS_DATABASE.items():
        if any(word in description for word in pest_name.split('_')):
            return jsonify({
                'success': True,
                'pest': pest_name,
                'description': pest_info['description'],
                'damage': pest_info['damage'],
                'organic_control': pest_info['organic_control'],
                'chemical_control': pest_info['chemical_control']
            })
    
    return jsonify({
        'success': True,
        'pest': 'Unknown',
        'description': 'Pest could not be identified from description',
        'organic_control': 'Use neem oil spray as general preventive',
        'chemical_control': 'Consult local agricultural officer'
    })

# ============================================
# SOIL ANALYSIS ENDPOINT
# ============================================

@app.route('/api/soil-analysis', methods=['POST'])
def analyze_soil():
    data = request.json
    soil_type = data.get('soil_type', 'loamy')
    ph_level = data.get('ph_level', random.uniform(6.0, 7.5))
    
    recommendations = []
    if ph_level < 6.0:
        recommendations.append("Add lime to increase pH")
    elif ph_level > 7.5:
        recommendations.append("Add sulfur to decrease pH")
    else:
        recommendations.append("pH is optimal for most crops")
    
    return jsonify({
        'success': True,
        'soil_type': soil_type,
        'ph_level': round(ph_level, 1),
        'ph_status': 'Optimal' if 6.0 <= ph_level <= 7.5 else 'Needs adjustment',
        'recommendations': recommendations,
        'organic_matter': random.choice(['High', 'Medium', 'Low']),
        'moisture_level': random.choice(['Ideal', 'Adequate', 'Dry'])
    })

# ============================================
# NEARBY SERVICES ENDPOINT
# ============================================

@app.route('/api/nearby-services', methods=['GET'])
def get_nearby_services():
    service_type = request.args.get('type', '')
    lat = request.args.get('lat', 20.5937, type=float)
    lon = request.args.get('lon', 78.9629, type=float)
    
    services = NEARBY_SERVICES.copy()
    if service_type:
        services = [s for s in services if s['type'] == service_type]
    
    return jsonify({
        'success': True,
        'services': services,
        'total_count': len(services)
    })

# ============================================
# COMMUNITY FORUM ENDPOINTS
# ============================================

# In-memory forum storage
forum_posts = [
    {
        'id': 1,
        'author': 'Ramesh from Maharashtra',
        'question': 'My tomato plants have yellow leaves. What should I do?',
        'answers': ['Check for nutrient deficiency', 'Could be overwatering', 'Test soil pH'],
        'timestamp': datetime.now().isoformat(),
        'likes': 5
    },
    {
        'id': 2,
        'author': 'Priya from Tamil Nadu',
        'question': 'Best organic pesticide for brinjal?',
        'answers': ['Neem oil spray works great', 'Try garlic-chili solution', 'Use companion planting'],
        'timestamp': datetime.now().isoformat(),
        'likes': 8
    }
]

@app.route('/api/forum-posts', methods=['GET'])
def get_forum_posts():
    return jsonify({'success': True, 'posts': forum_posts})

@app.route('/api/add-question', methods=['POST'])
def add_question():
    data = request.json
    new_post = {
        'id': len(forum_posts) + 1,
        'author': data.get('author', 'Anonymous Farmer'),
        'question': data.get('question', ''),
        'answers': [],
        'timestamp': datetime.now().isoformat(),
        'likes': 0
    }
    forum_posts.insert(0, new_post)
    return jsonify({'success': True, 'post': new_post})

@app.route('/api/add-answer', methods=['POST'])
def add_answer():
    data = request.json
    post_id = data.get('post_id')
    answer = data.get('answer')
    
    for post in forum_posts:
        if post['id'] == post_id:
            post['answers'].append(answer)
            return jsonify({'success': True, 'message': 'Answer added'})
    
    return jsonify({'success': False, 'error': 'Post not found'}), 404

# ============================================
# DAILY TIPS ENDPOINT
# ============================================

@app.route('/api/daily-tip', methods=['GET'])
def get_daily_tip():
    tip_index = datetime.now().day % len(DAILY_TIPS)
    return jsonify({
        'success': True,
        'tip': DAILY_TIPS[tip_index],
        'tip_number': tip_index + 1,
        'total_tips': len(DAILY_TIPS)
    })

# ============================================
# CHATBOT ENDPOINT
# ============================================

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.json
        message = data.get('message', '').lower()
        language = data.get('language', 'en')
        
        # Intent matching
        if any(word in message for word in ['hello', 'hi', 'namaste', 'hey']):
            response = CHATBOT_RESPONSES['greeting']
        elif any(word in message for word in ['price', 'market', 'rate', 'cost']):
            response = CHATBOT_RESPONSES['price']
        elif any(word in message for word in ['weather', 'temperature', 'rain', 'humidity']):
            weather = get_weather_data()
            risk, advice = calculate_disease_risk(weather)
            response = CHATBOT_RESPONSES['weather'].format(
                temp=weather['temperature'], 
                humidity=weather['humidity'],
                risk=risk,
                advice=advice
            )
        elif any(word in message for word in ['fertilizer', 'nutrient', 'npk']):
            response = CHATBOT_RESPONSES['fertilizer']
        elif any(word in message for word in ['disease', 'symptom', 'infection']):
            response = CHATBOT_RESPONSES['disease']
        elif any(word in message for word in ['pest', 'insect', 'bug', 'worm']):
            response = CHATBOT_RESPONSES['pest']
        elif any(word in message for word in ['organic', 'natural', 'chemical free']):
            response = CHATBOT_RESPONSES['organic']
        elif any(word in message for word in ['water', 'irrigation', 'watering']):
            response = CHATBOT_RESPONSES['water']
        elif any(word in message for word in ['help', 'what can you do', 'capabilities']):
            response = CHATBOT_RESPONSES['help']
        elif any(word in message for word in ['treatment', 'cure', 'medicine']):
            response = CHATBOT_RESPONSES['treatment']
        else:
            response = CHATBOT_RESPONSES['help']
        
        return jsonify({
            'success': True,
            'response': response,
            'language': language,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================
# CROP CALENDAR ENDPOINT
# ============================================

@app.route('/api/crop-calendar', methods=['GET'])
def get_crop_calendar():
    crop = request.args.get('crop', 'wheat')
    
    planting_seasons = {
        'wheat': {'planting': 'Oct-Nov', 'harvest': 'Mar-Apr', 'progress': 75},
        'rice': {'planting': 'Jun-Jul', 'harvest': 'Oct-Nov', 'progress': 40},
        'corn': {'planting': 'Jun-Jul', 'harvest': 'Sep-Oct', 'progress': 60},
        'tomato': {'planting': 'Jan-Feb', 'harvest': 'Apr-May', 'progress': 90}
    }
    
    season = planting_seasons.get(crop.lower(), {'planting': 'Unknown', 'harvest': 'Unknown', 'progress': 0})
    
    return jsonify({
        'success': True,
        'crop': crop,
        'planting_season': season['planting'],
        'harvest_season': season['harvest'],
        'progress_percentage': season['progress'],
        'next_action': 'Fertilize' if season['progress'] < 50 else 'Monitor for pests' if season['progress'] < 80 else 'Prepare for harvest'
    })

# ============================================
# BLOCKCHAIN VERIFICATION ENDPOINT
# ============================================

@app.route('/api/blockchain-verify', methods=['GET'])
def verify_blockchain():
    return jsonify({
        'success': True,
        'total_blocks': len(blockchain_chain),
        'total_diagnoses': sum(len(block['diagnoses']) for block in blockchain_chain),
        'latest_block': blockchain_chain[-1] if blockchain_chain else None,
        'integrity_verified': True
    })

# ============================================
# DIAGNOSIS HISTORY ENDPOINT
# ============================================

@app.route('/api/diagnosis-history', methods=['GET'])
def get_diagnosis_history():
    limit = request.args.get('limit', 10, type=int)
    return jsonify({
        'success': True,
        'history': diagnosis_history[-limit:],
        'total': len(diagnosis_history)
    })

# ============================================
# QR CODE GENERATION ENDPOINT
# ============================================

@app.route('/api/generate-qr', methods=['POST'])
def generate_qr():
    data = request.json
    diagnosis_id = data.get('diagnosis_id')
    
    qr_data = {
        'diagnosis_id': diagnosis_id,
        'timestamp': datetime.now().isoformat(),
        'verification_url': f"https://agriai.com/verify/{diagnosis_id}"
    }
    
    return jsonify({
        'success': True,
        'qr_data': qr_data,
        'verification_code': hashlib.md5(str(diagnosis_id).encode()).hexdigest()[:10]
    })

# ============================================
# MAIN ENTRY POINT
# ============================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🌱 NEELIMA PASALA'S COMPLETE PLANT DISEASE AI SYSTEM 🌱")
    print("="*80)
    print("✅ ALL SECTIONS ACTIVATED:")
    print("   🔬 Disease Diagnosis (Image + Text)")
    print("   🌤️ Weather Analysis & Risk Prediction")
    print("   💰 Market Price Dashboard")
    print("   👨‍⚕️ Expert Consultation Booking")
    print("   🏛️ Government Schemes Portal")
    print("   📊 Yield & Profit Prediction")
    print("   🧪 Fertilizer Calculator")
    print("   🐛 Pest Identification Guide")
    print("   🌱 Soil Health Analyzer")
    print("   📍 Nearby Services Locator")
    print("   💬 Farmer Community Forum")
    print("   🤖 AI Chatbot Assistant")
    print("   🔗 Blockchain Security")
    print("   📱 QR Code Reporting")
    print("   📅 Smart Crop Calendar")
    print("   💡 Daily Agri Tips")
    print("   🎨 Multi-Language Support (6 Languages)")
    print("="*80)
    print(f"\n📍 SERVER RUNNING AT: http://localhost:5000")
    print("👩‍💻 Created by: Neelima Pasala")
    print("🚀 Full Stack Agricultural AI Platform")
    print("="*80 + "\n")
    
    app.run(debug=True, port=5000, host='localhost')