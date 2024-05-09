import joblib
import numpy as np
classifier=joblib.load('diabetes.pkl')

scaler=joblib.load('scaler.pkl')

pregnancies = int(input("Enter the number of pregnancies: "))
glucose = float(input("Enter the glucose level: "))
blood_pressure = float(input("Enter the blood pressure: "))
skin_thickness = float(input("Enter the skin thickness: "))
insulin = float(input("Enter the insulin level: "))
bmi = float(input("Enter the BMI (Body Mass Index): "))
diabetes_pred = input("Enter the diabetes prediction (yes/no): ").lower() == "yes"
age = int(input("Enter the age: "))

        # Create a tuple with the collected data
user_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pred, age)
# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(user_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')