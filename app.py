import streamlit as st
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from datetime import datetime

# Load the trained model
model = load_model('my_model.keras')

# Define class names
class_names = ['Benign', 'Early', 'Pre', 'Pro']

# Define prescription templates for each class
prescription_templates = {
    'Benign': {
        'recommendations': [
            'Regular follow-up appointments every 6 months',
            'Routine blood tests',
            'Report any new symptoms immediately'
        ],
        'medications': [],
        'lifestyle': [
            'Maintain a healthy diet',
            'Regular exercise',
            'Adequate rest'
        ]
    },
    'Early': {
        'recommendations': [
            'Immediate consultation with an oncologist',
            'Complete blood count (CBC) test',
            'PET-CT scan',
            'Bone marrow biopsy may be required'
        ],
        'medications': [
            'To be prescribed by treating physician'
        ],
        'lifestyle': [
            'Balanced nutrition',
            'Moderate physical activity as tolerated',
            'Stress management techniques'
        ]
    },
    'Pre': {
        'recommendations': [
            'Urgent oncologist consultation',
            'Comprehensive staging workup',
            'Regular monitoring of blood counts',
            'Immunophenotyping'
        ],
        'medications': [
            'To be prescribed by treating physician'
        ],
        'lifestyle': [
            'Rest and recuperation',
            'Nutritional support',
            'Regular monitoring'
        ]
    },
    'Pro': {
        'recommendations': [
            'Immediate specialist intervention',
            'Comprehensive treatment plan',
            'Regular monitoring',
            'Support group participation'
        ],
        'medications': [
            'To be prescribed by treating physician'
        ],
        'lifestyle': [
            'Complete rest',
            'Supervised physical activity',
            'Nutritional counseling'
        ]
    }
}

def generate_html_report(predicted_class, confidence, date):
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Medical Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                max-width: 800px;
                margin: 0 auto;
            }}
            .header {{
                text-align: center;
                padding: 20px;
                background-color: #f8f9fa;
                margin-bottom: 20px;
                border-radius: 5px;
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .disclaimer {{
                font-style: italic;
                font-size: 0.9em;
                padding: 20px;
                background-color: #fff3cd;
                border-radius: 5px;
                margin-top: 30px;
            }}
            h1 {{
                color: #2c3e50;
                margin-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-bottom: 15px;
            }}
            ul {{
                padding-left: 20px;
            }}
            li {{
                margin-bottom: 10px;
            }}
            .ref-id {{
                color: #666;
                font-size: 0.9em;
                text-align: right;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Medical Report and Recommendations</h1>
            <p>Date: {date}</p>
        </div>

        <div class="section">
            <h2>Classification Results</h2>
            <p><strong>Predicted Class:</strong> {predicted_class}</p>
            <p><strong>Confidence Score:</strong> {confidence:.2f}%</p>
        </div>

        <div class="section">
            <h2>Recommended Actions</h2>
            <ul>
            {''.join(f'<li>{rec}</li>' for rec in prescription_templates[predicted_class]['recommendations'])}
            </ul>
        </div>

        <div class="section">
            <h2>Lifestyle Recommendations</h2>
            <ul>
            {''.join(f'<li>{item}</li>' for item in prescription_templates[predicted_class]['lifestyle'])}
            </ul>
        </div>

        <div class="disclaimer">
            <strong>IMPORTANT DISCLAIMER:</strong> This is an AI-generated report for informational purposes only. 
            This is not a medical diagnosis or prescription. Please consult with a qualified healthcare 
            professional for proper medical evaluation, diagnosis, and treatment.
        </div>

        <div class="ref-id">
            Reference ID: {datetime.now().strftime("%Y%m%d%H%M%S")}
        </div>
    </body>
    </html>
    """
    return html_content

# Streamlit UI
st.set_page_config(page_title="Lymphoma Classification", layout="centered")
st.title("Lymphoma Image Classification")
st.markdown("Upload an image to predict the class.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess image
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = img_array.reshape(1, 224, 224, 3)
    
    # Make prediction
    label = model.predict(img_array)
    predicted_class_index = np.argmax(label)
    predicted_class = class_names[predicted_class_index]
    confidence = float(label[0][predicted_class_index]) * 100
    
    # Display image and prediction
    st.image(uploaded_file, caption=f"Predicted Class: {predicted_class}", use_column_width=True)
    st.success(f"Prediction: {predicted_class}")
    
    # Show confidence scores
    st.subheader("Confidence Scores:")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {label[0][i]:.4f}")
    
    # Generate report
    if st.button("Generate Report"):
        html_content = generate_html_report(
            predicted_class,
            confidence,
            datetime.now().strftime("%Y-%m-%d")
        )
        
        st.download_button(
            label="Download Report (HTML)",
            data=html_content,
            file_name=f"medical_report_{datetime.now().strftime('%Y%m%d')}.html",
            mime="text/html"
        )