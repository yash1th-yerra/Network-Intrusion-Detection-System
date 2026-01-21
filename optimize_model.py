import tensorflow as tf
import os
import argparse

MODEL_PATH = "models/modified_bilstm_attention_model.h5"
OUTPUT_PATH = "models/nids_quantized.tflite"

def convert_to_tflite(model_path, output_path):
    print(f"Loading model from {model_path}...")
    # Load model (ignoring compilation to avoid custom object issues if not needed for inference)
    model = tf.keras.models.load_model(model_path, compile=False)
    
    print("Converting to TFLite with default optimization (quantization)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 8-bit quantization optimization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    print(f"Saving TFLite model to {output_path}...")
    with open(output_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"Original model size: {os.path.getsize(model_path) / 1024:.2f} KB")
    print(f"Quantized model size: {os.path.getsize(output_path) / 1024:.2f} KB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()
    
    convert_to_tflite(args.model, args.output)
