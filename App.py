# from flask import Flask, request, render_template
# import pandas as pd
# import joblib
# import numpy as np

# # Initialize Flask app
# app = Flask(__name__)

# # Load pre-trained models
# linear_model = joblib.load('linear_model.pkl')
# ridge_model = joblib.load('ridge_model.pkl')
# scaler = joblib.load('mpl_model.pkl')  # Used for neural network if needed

# # Create a dictionary to store models
# models = {
#     'linear': linear_model,
#     'ridge': ridge_model,
#     'neural': scaler
# }

# @app.route('/')
# def index():
#     # Render the homepage and pass model names to the form
#     return render_template('index.html', models=models.keys())

# @app.route('/predict', methods=['POST'])
# def predict():
#     selected_model_key = request.form.get('model')

#     # Get input data from form
#     input_data = [
#         float(request.form['ram']),
#         float(request.form['ssd']),
#         float(request.form['cpu']),
#         float(request.form['screen']),
#         float(request.form['battery']),
#         float(request.form['charging'])
#     ]
    
#     # Prepare input data for prediction
#     data = np.array(input_data).reshape(1, -1)

#     # if selected_model_key == 'neural':  # If using a neural network
#     #     data = .transform(data)

#     # Get the selected model
#     model = models[selected_model_key]

#     # Make predictions
#     prediction = model.predict(data)[0]

#     # Example reliability info (replace with actual calculated values)
#     reliability = {
#         "linear": 0.85,  # Example value, replace with your model's metrics
#         "ridge": 0.90,   # Example value
#         "neural": 0.92      # Example value if using neural network
#     }

#     # Render the result back to the template
#     return render_template('index.html', 
#                            prediction=prediction,
#                            models=models.keys(), 
#                            selected_model=selected_model_key, 
#                            reliability=reliability[selected_model_key])

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request
import pickle
import numpy as np

# Khởi tạo Flask app
app = Flask(__name__)

# Tải các mô hình đã huấn luyện
with open('linear_model.pkl', 'rb') as file:
    linear_model = pickle.load(file)

with open('ridge_model.pkl', 'rb') as file:
    ridge_model = pickle.load(file)

with open('nn_model.pkl', 'rb') as file:
    neural_model = pickle.load(file)


# Trang chủ hiển thị form nhập thông số
@app.route('/')
def index():
    return render_template('index.html')


# Xử lý dữ liệu sau khi form được submit
@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu từ form
    ram = float(request.form['ram'])
    ssd = float(request.form['ssd'])
    cpu = float(request.form['cpu'])
    screen_refresh_rate = float(request.form['screen_refresh_rate'])
    battery = float(request.form['battery'])
    charging_speed = float(request.form['charging_speed'])

    # Lấy mô hình mà người dùng đã chọn
    model_choice = request.form['model']

    # Tạo mảng dữ liệu đầu vào cho mô hình
    input_data = np.array([[ram, ssd, cpu, screen_refresh_rate, battery, charging_speed]])


    # Sử dụng mô hình tương ứng để dự đoán
    if model_choice == 'linear':
        prediction = linear_model.predict(input_data)
    elif model_choice == 'ridge':
        prediction = ridge_model.predict(input_data)
    elif model_choice == 'neural':
        prediction = neural_model.predict(input_data)/1000
    else:
        return 'Mô hình không hợp lệ!'

    # Trả về kết quả dự đoán
    return f'Giá dự đoán cho laptop là: {prediction[0]:.4f}'


# Chạy ứng dụng Flask
if __name__ == '__main__':
    app.run(debug=True)