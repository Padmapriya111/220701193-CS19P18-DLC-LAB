# cheque_digit_processing.py
import os
import cv2
import numpy as np
import pandas as pd
from datetime import date
from tensorflow.keras.models import load_model

def update_cheque_record(cheque_number, account_number,amount_digits=None):
    csv_path = "outputs/cheque_records.csv"
    os.makedirs("outputs", exist_ok=True)
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if cheque_number in df['cheque_number'].values:
            idx = df.index[df['cheque_number'] == cheque_number][0]
            if amount_digits is not None:
                df.at[idx, 'amount_digits'] = amount_digits
        else:
            df = pd.concat([df, pd.DataFrame([{
                'cheque_number': cheque_number,
                'account_number': account_number,
                'amount_digits': amount_digits or "",
                'date_cleared': str(date.today())
            }])], ignore_index=True)
    else:
        df = pd.DataFrame([{
            'cheque_number': cheque_number,
            'account_number': account_number,
            'amount_digits': amount_digits or "",
            'date_cleared': str(date.today())
        }])
    
    df.to_csv(csv_path, index=False)
    print(f"Cheque record updated in {csv_path}")
model_path = "models/cheque_digit_cnn_model.h5"
model = load_model(model_path)
print("Digit CNN model loaded successfully!")
digit_folder = "digits/"
digit_images = sorted(os.listdir(digit_folder))
predicted_digits = []

for img_file in digit_images:
    img_path = os.path.join(digit_folder, img_file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28,28))
    # Threshold & invert to MNIST-like style
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    # Normalize
    img = img / 255.0
    # Reshape for CNN
    img = img.reshape(1,28,28,1)
    pred = model.predict(img)
    predicted_digits.append(str(np.argmax(pred)))

cheque_amount = "".join(predicted_digits)
print("Predicted Cheque Amount:", cheque_amount)
update_cheque_record(
    cheque_number="102345",
    account_number="1234567890",
    amount_digits=cheque_amount
)
