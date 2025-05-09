import os
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image
import streamlit as st

def generate_confusion_matrix(y_true, y_pred, model_name=""):
    """Generate and save confusion matrix"""
    # Create temp directory (works in Streamlit Cloud)
    temp_dir = tempfile.mkdtemp()
    cm_path = os.path.join(temp_dir, f"confusion_matrix_{model_name}.png")
    
    # Generate matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Save to temp file
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    
    return cm_path

def show_evaluation(y_true, y_pred, model_name=""):
    """Display evaluation metrics in Streamlit"""
    try:
        # Generate and display confusion matrix
        cm_path = generate_confusion_matrix(y_true, y_pred, model_name)
        
        st.subheader(f"Model Evaluation: {model_name}")
        st.image(cm_path)
        
        # Clean up (important for Streamlit Cloud)
        if os.path.exists(cm_path):
            os.remove(cm_path)
            
    except Exception as e:
        st.error(f"Evaluation error: {str(e)}")
