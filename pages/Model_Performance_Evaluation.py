import os
import tempfile
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, model_name=""):
    """Generate and display confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Save to temporary file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(temp_path, bbox_inches='tight')
    plt.close()
    
    return temp_path

def show_evaluation(y_true, y_pred, model_name=""):
    """Display model evaluation in Streamlit"""
    # Generate confusion matrix
    try:
        cm_path = plot_confusion_matrix(y_true, y_pred, model_name)
        
        # Display in Streamlit
        st.subheader(f"Model Evaluation: {model_name}")
        st.image(cm_path)
        
        # Clean up temporary file
        if os.path.exists(cm_path):
            os.remove(cm_path)
            
    except Exception as e:
        st.error(f"Error generating evaluation: {str(e)}")
