# ============================================================================
# VISHING FORENSICS SUITE - MAIN APPLICATION
# File: app.py
# Author: Benjamen Elungu
# Institution: Botho University
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timedelta
import random

# Ensure proper imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for the application"""
    APP_TITLE = "AI-Powered Vishing Forensics Suite"
    APP_VERSION = "1.0.0"
    
    # Paths
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(DATA_DIR, "model")
    SCENARIOS_DB = os.path.join(DATA_DIR, "scenarios_database.csv")
    CASES_DB = os.path.join(DATA_DIR, "cases_database.csv")
    
    # Simulation parameters
    GOVERNMENT_DEPARTMENTS = [
        "Ministry of Finance", "Inland Revenue", "Social Security",
        "Immigration Services", "Police Department", "Municipal Council",
        "Ministry of Health", "Bank of Namibia"
    ]
    
    SCAM_TYPES = [
        "Tax Refund", "Social Grant", "License Renewal",
        "Debt Collection", "Prize Winning", "Account Verification",
        "COVID-19 Relief", "Pension Payment"
    ]
    
    URGENCY_LEVELS = ["Low", "Medium", "High", "Critical"]

# ============================================================================
# DATA INITIALIZATION
# ============================================================================

def initialize_data_structure():
    """Create necessary directories and files"""
    config = Config()
    
    # Create directories
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # Initialize scenarios database
    if not os.path.exists(config.SCENARIOS_DB):
        scenarios_df = pd.DataFrame(columns=[
            'scenario_id', 'timestamp', 'department', 'scam_type',
            'urgency', 'script', 'metadata', 'suspicious_score'
        ])
        scenarios_df.to_csv(config.SCENARIOS_DB, index=False)
    
    # Initialize cases database
    if not os.path.exists(config.CASES_DB):
        cases_df = pd.DataFrame(columns=[
            'case_id', 'timestamp', 'phone_number', 'call_duration',
            'call_frequency', 'risk_score', 'status'
        ])
        cases_df.to_csv(config.CASES_DB, index=False)

# ============================================================================
# SCENARIO GENERATOR MODULE
# ============================================================================

class ScenarioGenerator:
    """Vishing scenario simulation engine"""
    
    def __init__(self):
        self.config = Config()
        self.load_data()
    
    def load_data(self):
        """Load existing scenarios"""
        try:
            self.scenarios_df = pd.read_csv(self.config.SCENARIOS_DB)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            self.scenarios_df = pd.DataFrame(columns=[
                'scenario_id', 'timestamp', 'department', 'scam_type',
                'urgency', 'script', 'metadata', 'suspicious_score'
            ])
    
    def generate_script(self, department, scam_type, urgency):
        """Generate realistic vishing script"""
        
        scripts = {
            "Tax Refund": [
                f"Good day from {department}. We are processing tax refunds for 2023. You are eligible for NAD 5,800. To expedite processing, we need to verify your banking details.",
                f"Hello, this is {department} Tax Division. Your account shows an overpayment of NAD 4,200. For immediate refund processing, please confirm your account information.",
                f"{department} calling. We have urgent tax refund notification. Amount: NAD 6,150. This expires in 48 hours. Please provide verification."
            ],
            "Social Grant": [
                f"This is {department}. Your social grant payment has been suspended due to a system error. To reinstate, we need your ID number immediately.",
                f"Good afternoon from {department}. We are increasing social grants by 15%. To register for the increase, confirm your personal details now.",
                f"{department} here. There is a problem with your grant account. Without immediate action, payments will stop."
            ],
            "License Renewal": [
                f"{department} notification: Your license expires tomorrow. Pay renewal fee of NAD 350 via mobile money to avoid penalties.",
                f"This is {department}. Your vehicle registration has lapsed. Immediate payment required to avoid court summons.",
                f"Urgent from {department}: License renewal overdue. Pay NAD 450 now to prevent license suspension."
            ],
            "Debt Collection": [
                f"{department} Legal Division calling. You have unpaid municipal fees of NAD 2,800. Legal action begins in 24 hours unless settled.",
                f"This is {department} Collections. Outstanding debt of NAD 3,500. Warrant of arrest issued. Call this number to resolve.",
                f"{department} notice: Failure to pay NAD 4,200 by end of day will result in asset seizure."
            ],
            "Prize Winning": [
                f"Congratulations from {department}! You won NAD 50,000 in the National Lottery. Processing fee of NAD 500 required to claim prize.",
                f"{department} Prize Department: You are selected for NAD 75,000 government grant. Pay administration fee to receive funds.",
                f"Great news! {department} confirms you won NAD 100,000. Send NAD 800 for tax clearance to release your winnings."
            ],
            "Account Verification": [
                f"{department} Security Team: Suspicious activity on your account. Verify identity by providing your PIN and ID number.",
                f"This is {department}. Your account will be frozen unless you confirm details within one hour.",
                f"{department} calling. System upgrade requires you to verify your banking information immediately."
            ],
            "COVID-19 Relief": [
                f"{department} COVID Relief: You qualify for NAD 3,000 emergency fund. Provide banking details for direct deposit.",
                f"This is {department}. Pandemic relief payment of NAD 4,500 approved. Confirm account to receive funds today.",
                f"{department} notification: Final day to claim NAD 5,000 COVID assistance. Act now or lose eligibility."
            ],
            "Pension Payment": [
                f"{department} Pensions: Your monthly payment of NAD 1,800 is delayed. Update details to release funds.",
                f"This is {department}. Pension system upgrade requires re-registration. Provide ID and bank account.",
                f"{department} calling: Pension arrears of NAD 12,000 available. Processing fee required."
            ]
        }
        
        base_script = random.choice(scripts.get(scam_type, [
            f"This is {department}. We need to verify your information for official purposes."
        ]))
        
        # Add urgency modifiers
        if urgency == "High":
            urgency_phrases = [
                " This is URGENT - respond within 2 hours!",
                " IMMEDIATE action required or account will be suspended!",
                " This is your FINAL NOTICE before legal proceedings!"
            ]
            base_script += random.choice(urgency_phrases)
        elif urgency == "Critical":
            base_script = "**URGENT** " + base_script + " **ACT NOW OR FACE CONSEQUENCES**"
        
        return base_script
    
    def generate_metadata(self, urgency):
        """Generate realistic call metadata"""
        base_duration = 120  # 2 minutes average
        
        if urgency in ["High", "Critical"]:
            duration = random.randint(180, 420)  # 3-7 minutes
            frequency = random.randint(5, 15)
        else:
            duration = random.randint(60, 180)
            frequency = random.randint(1, 5)
        
        return {
            'call_duration': duration,
            'time_of_day': random.choice(['Morning', 'Afternoon', 'Evening']),
            'caller_id_spoofed': random.choice([True, False]),
            'call_frequency_per_day': frequency,
            'requested_info': random.sample([
                'bank_details', 'id_number', 'password', 
                'pin_code', 'address', 'phone_number'
            ], k=random.randint(1, 3))
        }
    
    def calculate_suspicious_score(self, scam_type, urgency, metadata):
        """Calculate suspiciousness score 0-100"""
        score = 0
        
        # Scam type weighting
        high_risk_types = ["Debt Collection", "Prize Winning", "Account Verification"]
        if scam_type in high_risk_types:
            score += 30
        else:
            score += 15
        
        # Urgency weighting
        urgency_scores = {"Low": 10, "Medium": 20, "High": 35, "Critical": 45}
        score += urgency_scores.get(urgency, 10)
        
        # Metadata factors
        if metadata['call_duration'] > 300:
            score += 15
        if metadata['call_frequency_per_day'] > 5:
            score += 10
        if metadata['caller_id_spoofed']:
            score += 10
        if len(metadata['requested_info']) >= 3:
            score += 10
        
        return min(score, 100)
    
    def save_scenario(self, scenario_data):
        """Save scenario to database"""
        new_row = pd.DataFrame([scenario_data])
        self.scenarios_df = pd.concat([self.scenarios_df, new_row], ignore_index=True)
        self.scenarios_df.to_csv(self.config.SCENARIOS_DB, index=False)
    
    def show_interface(self):
        """Display scenario generator interface"""
        st.title("Vishing Scenario Simulator")
        st.markdown("Generate realistic Namibian vishing scenarios for training and testing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Scenario Parameters")
            department = st.selectbox("Government Department", self.config.GOVERNMENT_DEPARTMENTS)
            scam_type = st.selectbox("Scam Type", self.config.SCAM_TYPES)
            urgency = st.select_slider("Urgency Level", self.config.URGENCY_LEVELS)
            num_scenarios = st.number_input("Number of Scenarios", 1, 50, 5)
        
        with col2:
            st.subheader("Quick Stats")
            st.metric("Total Scenarios Generated", len(self.scenarios_df))
            if not self.scenarios_df.empty:
                avg_score = self.scenarios_df['suspicious_score'].mean()
                st.metric("Average Suspicious Score", f"{avg_score:.1f}/100")
        
        if st.button("Generate Scenarios", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            generated_scenarios = []
            
            for i in range(num_scenarios):
                status_text.text(f"Generating scenario {i+1}/{num_scenarios}...")
                
                scenario_id = f"SCN_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:03d}"
                script = self.generate_script(department, scam_type, urgency)
                metadata = self.generate_metadata(urgency)
                suspicious_score = self.calculate_suspicious_score(scam_type, urgency, metadata)
                
                scenario_data = {
                    'scenario_id': scenario_id,
                    'timestamp': datetime.now().isoformat(),
                    'department': department,
                    'scam_type': scam_type,
                    'urgency': urgency,
                    'script': script,
                    'metadata': json.dumps(metadata),
                    'suspicious_score': suspicious_score
                }
                
                self.save_scenario(scenario_data)
                generated_scenarios.append(scenario_data)
                
                progress_bar.progress((i + 1) / num_scenarios)
            
            status_text.text("Generation complete!")
            st.success(f"Successfully generated {num_scenarios} scenarios!")
            
            # Display latest scenario
            if generated_scenarios:
                st.subheader("Latest Generated Scenario")
                latest = generated_scenarios[-1]
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.text_area("Script", latest['script'], height=150)
                with col2:
                    st.metric("Risk Score", f"{latest['suspicious_score']}/100")
                    st.write(f"**Type:** {latest['scam_type']}")
                    st.write(f"**Urgency:** {latest['urgency']}")
                
                with st.expander("View Metadata"):
                    st.json(json.loads(latest['metadata']))

# ============================================================================
# PATTERN RECOGNITION MODULE
# ============================================================================

class VishingDetector:
    """Machine learning-based pattern recognition"""
    
    def __init__(self):
        self.config = Config()
        self.model = None
        self.model_trained = False
    
    def generate_training_data(self, num_samples=2000):
        """Generate synthetic training dataset"""
        np.random.seed(42)
        data = []
        
        for i in range(num_samples):
            is_vishing = 1 if i >= num_samples // 2 else 0
            
            if is_vishing:
                # Vishing call patterns
                call_duration = np.random.normal(300, 100)
                call_frequency = np.random.poisson(8)
                time_of_day = random.randint(9, 20)  # Business hours
                urgency_level = random.randint(3, 5)
            else:
                # Legitimate call patterns
                call_duration = np.random.normal(120, 40)
                call_frequency = np.random.poisson(2)
                time_of_day = random.randint(0, 23)
                urgency_level = random.randint(1, 3)
            
            data.append([
                max(10, call_duration),
                max(1, call_frequency),
                time_of_day,
                urgency_level,
                is_vishing
            ])
        
        return pd.DataFrame(data, columns=[
            'call_duration', 'call_frequency', 'time_of_day', 
            'urgency_level', 'is_vishing'
        ])
    
    def train_model(self):
        """Train Random Forest classifier"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        import joblib
        
        # Generate training data
        data = self.generate_training_data(2000)
        X = data.drop('is_vishing', axis=1)
        y = data['is_vishing']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model
        model_path = os.path.join(self.config.MODEL_DIR, 'vishing_classifier.pkl')
        joblib.dump(self.model, model_path)
        
        self.model_trained = True
        
        return accuracy, X_test, y_test, y_pred
    
    def load_model(self):
        """Load trained model"""
        import joblib
        model_path = os.path.join(self.config.MODEL_DIR, 'vishing_classifier.pkl')
        
        try:
            self.model = joblib.load(model_path)
            self.model_trained = True
            return True
        except FileNotFoundError:
            return False
    
    def predict(self, features):
        """Predict if call is vishing"""
        if not self.model_trained or self.model is None:
            return None
        
        prediction = self.model.predict([features])[0]
        probability = self.model.predict_proba([features])[0]
        
        return {
            'prediction': 'Vishing' if prediction == 1 else 'Legitimate',
            'confidence': max(probability) * 100,
            'vishing_probability': probability[1] * 100,
            'legitimate_probability': probability[0] * 100
        }
    
    def show_interface(self):
        """Display pattern recognition interface"""
        st.title("Pattern Recognition System")
        st.markdown("AI-powered detection of vishing call patterns")
        
        tab1, tab2 = st.tabs(["Model Training", "Call Analysis"])
        
        with tab1:
            st.subheader("Train Vishing Detection Model")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Train New Model", type="primary"):
                    with st.spinner("Training model... This may take 1-2 minutes"):
                        accuracy, X_test, y_test, y_pred = self.train_model()
                    
                    st.success(f"Model trained successfully!")
                    st.metric("Model Accuracy", f"{accuracy:.1%}")
                    
                    # Feature importance
                    importance_df = pd.DataFrame({
                        'Feature': X_test.columns,
                        'Importance': self.model.feature_importances_
                    }).sort_values('Importance', ascending=True)
                    
                    st.bar_chart(importance_df.set_index('Feature'))
            
            with col2:
                model_loaded = self.load_model()
                if model_loaded:
                    st.success("Model loaded successfully")
                    st.info("Model is ready for predictions")
                else:
                    st.warning("No trained model found")
                    st.info("Please train a new model first")
        
        with tab2:
            st.subheader("Analyze Single Call")
            
            if not self.load_model():
                st.error("Please train the model first in the 'Model Training' tab")
                return
            
            col1, col2 = st.columns(2)
            
            with col1:
                call_duration = st.number_input(
                    "Call Duration (seconds)", 
                    min_value=10, 
                    max_value=3600, 
                    value=180
                )
                call_frequency = st.number_input(
                    "Calls per Day", 
                    min_value=1, 
                    max_value=50, 
                    value=3
                )
            
            with col2:
                time_of_day = st.slider("Time of Day (24h)", 0, 23, 12)
                urgency_level = st.slider("Urgency Level (1-5)", 1, 5, 3)
            
            if st.button("Analyze Call", type="primary"):
                features = [call_duration, call_frequency, time_of_day, urgency_level]
                result = self.predict(features)
                
                st.subheader("Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Prediction", result['prediction'])
                with col2:
                    st.metric("Confidence", f"{result['confidence']:.1f}%")
                with col3:
                    risk_level = "HIGH" if result['vishing_probability'] > 70 else "MEDIUM" if result['vishing_probability'] > 40 else "LOW"
                    st.metric("Risk Level", risk_level)
                
                # Probability breakdown
                prob_data = pd.DataFrame({
                    'Type': ['Legitimate', 'Vishing'],
                    'Probability': [result['legitimate_probability'], result['vishing_probability']]
                })
                st.bar_chart(prob_data.set_index('Type'))

# ============================================================================
# NLP INVESTIGATION ASSISTANT
# ============================================================================

class NLPAssistant:
    """Natural language processing for transcript analysis"""
    
    def __init__(self):
        self.config = Config()
        
        # Vishing indicators
        self.keywords = {
            'urgency': ['urgent', 'immediately', 'now', 'today', 'deadline', 'expires', 'final'],
            'financial': ['money', 'payment', 'refund', 'account', 'bank', 'transfer', 'PIN'],
            'authority': ['government', 'ministry', 'revenue', 'police', 'official', 'legal'],
            'threat': ['suspend', 'close', 'arrest', 'penalty', 'fine', 'legal action'],
            'request': ['verify', 'confirm', 'provide', 'send', 'give us']
        }
    
    def analyze_text(self, text):
        """Analyze transcript for vishing indicators"""
        text_lower = text.lower()
        
        # Find keywords
        found_keywords = {}
        for category, words in self.keywords.items():
            found = [w for w in words if w in text_lower]
            if found:
                found_keywords[category] = found
        
        # Calculate risk score
        risk_score = 0
        risk_score += len(found_keywords.get('urgency', [])) * 15
        risk_score += len(found_keywords.get('financial', [])) * 10
        risk_score += len(found_keywords.get('authority', [])) * 10
        risk_score += len(found_keywords.get('threat', [])) * 20
        risk_score += len(found_keywords.get('request', [])) * 12
        
        risk_score = min(risk_score, 100)
        
        # Generate recommendations
        recommendations = []
        if risk_score > 70:
            recommendations.append("HIGH RISK: Strong vishing indicators detected. Recommend immediate investigation.")
        elif risk_score > 40:
            recommendations.append("MEDIUM RISK: Multiple suspicious elements. Further analysis recommended.")
        else:
            recommendations.append("LOW RISK: Few indicators, but verify through official channels.")
        
        if 'authority' in found_keywords:
            recommendations.append("Verify caller identity through official department contacts")
        if 'financial' in found_keywords:
            recommendations.append("Never provide banking information without independent verification")
        if 'urgency' in found_keywords:
            recommendations.append("High-pressure tactics detected - take time to verify claims")
        
        return {
            'risk_score': risk_score,
            'keywords_found': found_keywords,
            'recommendations': recommendations,
            'word_count': len(text.split()),
            'sentence_count': len([s for s in text.split('.') if s.strip()])
        }
    
    def show_interface(self):
        """Display NLP assistant interface"""
        st.title("Intelligent Investigation Assistant")
        st.markdown("NLP-powered analysis of call transcripts")
        
        # Input method selection
        input_method = st.radio(
            "Input Method",
            ["Paste Transcript", "Use Generated Scenario", "Upload File"]
        )
        
        text_to_analyze = ""
        
        if input_method == "Paste Transcript":
            text_to_analyze = st.text_area(
                "Paste call transcript here:",
                height=200,
                placeholder="Enter the conversation transcript..."
            )
        
        elif input_method == "Use Generated Scenario":
            try:
                scenarios_df = pd.read_csv(self.config.SCENARIOS_DB)
                if not scenarios_df.empty:
                    scenario_options = [
                        f"{row['scenario_id']} - {row['scam_type']}" 
                        for _, row in scenarios_df.tail(10).iterrows()
                    ]
                    selected = st.selectbox("Select Scenario", scenario_options)
                    selected_id = selected.split(' - ')[0]
                    text_to_analyze = scenarios_df[
                        scenarios_df['scenario_id'] == selected_id
                    ]['script'].values[0]
                    st.text_area("Scenario Script:", text_to_analyze, height=150)
                else:
                    st.warning("No scenarios found. Generate some first.")
            except FileNotFoundError:
                st.error("Scenarios database not found")
        
        if st.button("Analyze Transcript", type="primary") and text_to_analyze.strip():
            with st.spinner("Analyzing transcript..."):
                analysis = self.analyze_text(text_to_analyze)
            
            # Results
            st.subheader("Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Score", f"{analysis['risk_score']}/100")
            with col2:
                st.metric("Word Count", analysis['word_count'])
            with col3:
                risk_level = "HIGH" if analysis['risk_score'] > 70 else "MEDIUM" if analysis['risk_score'] > 40 else "LOW"
                st.metric("Risk Level", risk_level)
            
            # Keywords found
            if analysis['keywords_found']:
                st.subheader("Vishing Indicators Detected")
                for category, keywords in analysis['keywords_found'].items():
                    st.write(f"**{category.title()}:** {', '.join(keywords)}")
            
            # Recommendations
            st.subheader("Investigation Recommendations")
            for rec in analysis['recommendations']:
                st.info(rec)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class VishingForensicsApp:
    """Main application controller"""
    
    def __init__(self):
        self.config = Config()
        initialize_data_structure()
        
        self.scenario_generator = ScenarioGenerator()
        self.vishing_detector = VishingDetector()
        self.nlp_assistant = NLPAssistant()
    
    def show_dashboard(self):
        """Display main dashboard"""
        st.title("Vishing Forensics Suite")
        st.markdown("AI-Powered Digital Forensics for Combating Vishing in Namibia")
        
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        
        scenarios_count = len(self.scenario_generator.scenarios_df)
        
        with col1:
            st.metric("Scenarios Generated", scenarios_count)
        with col2:
            model_status = "Ready" if self.vishing_detector.load_model() else "Not Trained"
            st.metric("ML Model Status", model_status)
        with col3:
            st.metric("System Version", self.config.APP_VERSION)
        with col4:
            st.metric("Active Modules", "3/3")
        
        # Quick start guide
        st.subheader("Quick Start Guide")
        st.markdown("""
        1. **Generate Scenarios**: Create realistic vishing scenarios for training
        2. **Train Model**: Build ML model for pattern recognition
        3. **Analyze Calls**: Use NLP to analyze call transcripts
        """)
        
        # System info
        with st.expander("System Information"):
            st.write(f"**Data Directory:** {self.config.DATA_DIR}")
            st.write(f"**Model Directory:** {self.config.MODEL_DIR}")
            st.write(f"**Python Version:** {sys.version.split()[0]}")
    
    def run(self):
        """Run the application"""
        st.set_page_config(
            page_title=self.config.APP_TITLE,
            page_icon="🔒",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        st.sidebar.markdown("---")
        
        page = st.sidebar.radio(
            "Select Module",
            [
                "Dashboard",
                "Scenario Simulator",
                "Pattern Recognition",
                "Investigation Assistant",
                "About"
            ]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.info(f"Version {self.config.APP_VERSION}")
        
        # Route to selected page
        if page == "Dashboard":
            self.show_dashboard()
        elif page == "Scenario Simulator":
            self.scenario_generator.show_interface()
        elif page == "Pattern Recognition":
            self.vishing_detector.show_interface()
        elif page == "Investigation Assistant":
            self.nlp_assistant.show_interface()
        elif page == "About":
            self.show_about()
    
    def show_about(self):
        """Display about page"""
        st.title("About This System")
        
        st.markdown("""
        ## AI-Powered Vishing Forensics Suite
        
        **Developer:** Benjamen Elungu  
        **Institution:** Botho University  
        **Programme:** Bachelor (Honours) of Science in Network Security and Computer Forensics
        
        ### Purpose
        This system addresses the critical need for advanced digital forensics tools to combat
        voice phishing (vishing) attacks in Namibia. It combines artificial intelligence,
        machine learning, and natural language processing to assist law enforcement in
        investigating and preventing vishing crimes.
        
        ### Key Features
        - **Scenario Simulation**: Generate realistic vishing scenarios for training
        - **Pattern Recognition**: ML-based detection of suspicious call patterns
        - **NLP Analysis**: Intelligent analysis of call transcripts
        
        ### Technology Stack
        - Python 3.8+
        - Streamlit (Web Framework)
        - Scikit-learn (Machine Learning)
        - Pandas & NumPy (Data Processing)
        
        ### Research Objectives
        1. Develop functional vishing simulation platform
        2. Engineer robust pattern recognition system (F1-score ≥ 0.85)
        3. Create intuitive investigation assistant
        4. Integrate components into cohesive web application
        
        ### Contact
        For questions or feedback about this research project, please contact your supervisor.
        """)

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    app = VishingForensicsApp()
    app.run()