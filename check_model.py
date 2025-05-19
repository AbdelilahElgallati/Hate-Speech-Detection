import h5py
import os
from huggingface_hub import hf_hub_download
from config import HF_TOKEN, HF_REPO_ID
import tensorflow as tf

def check_model_file():
    try:
        # Download model file
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename='model_hate_speech.h5',
            token=HF_TOKEN
        )
        
        print(f"Model path: {model_path}")
        print(f"File exists: {os.path.exists(model_path)}")
        print(f"File size: {os.path.getsize(model_path)} bytes")
        
        # Try to open the file with h5py
        try:
            with h5py.File(model_path, 'r') as f:
                print("\nModel file structure:")
                print("Keys:", list(f.keys()))
                
                # Print model configuration
                if 'model_config' in f:
                    print("\nModel config:", f['model_config'][()])
                
                # Print layer information
                if 'model_weights' in f:
                    print("\nModel weights structure:")
                    for key in f['model_weights'].keys():
                        print(f"- {key}")
        except Exception as h5py_error:
            print(f"\nError opening with h5py: {str(h5py_error)}")
            
            # Try to load with TensorFlow
            try:
                print("\nAttempting to load with TensorFlow...")
                model = tf.keras.models.load_model(model_path, compile=False)
                print("Successfully loaded with TensorFlow!")
                print(f"Model summary:\n{model.summary()}")
            except Exception as tf_error:
                print(f"Error loading with TensorFlow: {str(tf_error)}")
                
    except Exception as e:
        print(f"Error checking model file: {str(e)}")

if __name__ == "__main__":
    print(f"TensorFlow version: {tf.__version__}")
    check_model_file() 