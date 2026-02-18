# Indian Food Classifier App

This is a deep learning-based image classification application that identifies various Indian and popular food items from uploaded images.

## Model Capabilities

The model is trained to classify **20 different food types**:

### Indian Dishes
- **Breads & Staples**: Butter Naan, Chapati, Dhokla, Idli
- **Curries & Main Courses**: Chole Bhature, Dal Makhani, Kadai Paneer, Masala Dosa
- **Street Food & Appetizers**: Kaathi Rolls, Pakode, Paani Puri, Samosa
- **Desserts & Beverages**: Chai, Jalebi, Kulfi

### Other Foods
- Burger, Fried Rice, Momos, Pav Bhaji, Pizza

## Supported Image Formats

The app supports the following image file formats:
- PNG (`.png`)
- JPEG (`.jpg`, `.jpeg`)

## Model Architecture

- **Base Model**: MobileNetV2 (efficient and lightweight)
- **Input Size**: 224 x 224 pixels (RGB)
- **Output**: Probability distribution across 20 food classes

## How to Use

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Upload an image of food in PNG or JPEG format

3. The model will predict the food type and display:
   - The predicted food name
   - Confidence score visualization (bar chart showing probabilities for all classes)

## Requirements

See `requirements.txt` for dependencies. The model requires:
- TensorFlow
- Streamlit
- PIL (Pillow)
- NumPy

## Notes

- The model preprocesses images to 224x224 RGB format
- Results are most accurate for clear, well-lit images of the food items
- Predictions are provided with a confidence distribution across all 20 food classes
