from django.shortcuts import render,redirect,redirect,get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from django.urls import reverse
from django.contrib.auth import authenticate, login, logout
from django.db import IntegrityError
from django.contrib.auth.hashers import check_password

from .models import PatientInfo,SurveyResults

import os
import cv2
import ast
import joblib
import numpy as np
import tensorflow as tf

import requests
from io import BytesIO

def index(request):
    return render(request,'SplashScreen.html')

def survey(request):
    return render(request,'Survey.html')

def process_survey(request):
    if request.method == 'POST':
        NAME = request.POST.get('name', '1')
        GENDER = request.POST.get('gender', '1')
        AGE = request.POST.get('age', '1')
        SMOKING = request.POST.get('smoking', '1')
        YELLOW_FINGERS = request.POST.get('yellow-fingers', '1')
        ANXIETY = request.POST.get('anxiety', '1')
        PEER_PRESSURE = request.POST.get('peer-pressure', '1')
        CHRONIC_DISEASE = request.POST.get('chronic-disease', '1')
        FATIGUE = request.POST.get('fatigue', '1')
        ALLERGY = request.POST.get('allergies', '1')
        WHEEZING = request.POST.get('wheezing', '1')
        ALCOHOL_CONSUMING = request.POST.get('alcohol-consumption', '1')
        COUGHING = request.POST.get('coughing', '1')
        SHORTNESS_OF_BREATH = request.POST.get('shortness-of-breath', '1')
        SWALLOWING_DIFFICULTY = request.POST.get('swallowing-difficulty', '1')
        CHEST_PAIN = request.POST.get('chest-pain', '1')

        patient_info = PatientInfo.objects.create(
            name=NAME,
            age=AGE,
            gender= 'Male' if eval(GENDER) else 'Female'
        )
        patient_info.save()

        data = {
            "GENDER": GENDER,
            "AGE": AGE,
            "SMOKING": SMOKING,
            "YELLOW_FINGERS": YELLOW_FINGERS,
            "ANXIETY": ANXIETY,
            "PEER_PRESSURE": PEER_PRESSURE,
            "CHRONIC_DISEASE": CHRONIC_DISEASE,
            "FATIGUE": FATIGUE,
            "ALLERGY": ALLERGY,
            "WHEEZING": WHEEZING,
            "ALCOHOL_CONSUMING": ALCOHOL_CONSUMING,
            "COUGHING": COUGHING,
            "SHORTNESS_OF_BREATH": SHORTNESS_OF_BREATH,
            "SWALLOWING_DIFFICULTY": SWALLOWING_DIFFICULTY,
            "CHEST_PAIN": CHEST_PAIN
        }
        
        input_data = [[data[field] for field in data.keys()]]

        model = joblib.load(settings.TEXT_MODEL_PATH)  # Load the model

        predictions = model.predict(input_data) # Prediction

        probabilities = model.predict_proba(input_data) # Percentage of Prediction
        
        prediction_label = f"{predictions}\t{probabilities[0][1]*100}\t{AGE}"

        survey_results = SurveyResults.objects.create(
            patient_id=patient_info,
            smoking=SMOKING,
            yellow_fingers=YELLOW_FINGERS,
            anxiety=ANXIETY,
            peer_pressure=PEER_PRESSURE,
            chronic_disease=CHRONIC_DISEASE,
            fatigue=FATIGUE,
            allergy=ALLERGY,
            wheezing=WHEEZING,
            alcohol_consuming=ALCOHOL_CONSUMING,
            coughing=COUGHING,
            shortness_of_breath=SHORTNESS_OF_BREATH,
            swallowing_difficulty=SWALLOWING_DIFFICULTY,
            chest_pain=CHEST_PAIN,
            text_prediction=predictions[0],
            text_prediction_percentage=probabilities[0][1]*100
        )
        survey_results.save()

        return redirect('success')

def process_image(request):
    
    model = tf.keras.models.load_model(settings.IMAGE_MODEL_PATH)

    if request.method == 'POST' and 'image' in request.FILES:
        uploaded_image = request.FILES['image']
        patient_id = request.POST.get('patient_id')

        # Read the image
        img = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

        # Resize the image to match the input size of the model
        img_resized = cv2.resize(img, (224, 224))

        # Convert the image to the format expected by the model (e.g., convert to RGB and normalize)
        img_preprocessed = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) / 255.0

        # Expand dimensions to match the input shape of the model
        img_preprocessed = np.expand_dims(img_preprocessed, axis=0)

        # Make prediction using the model
        prediction = model.predict(img_preprocessed)

        class_labels = ["Malignant", "Normal", "Benign"]

        # Format the prediction result
        formatted_prediction = [(label, prob) for label, prob in zip(class_labels, prediction[0])]

        try:
            # Retrieve the existing SurveyResults record for the given patient_id
            survey_results = SurveyResults.objects.get(patient_id=patient_id)
            
            # Update the ct_prediction field with the new formatted_prediction value
            survey_results.ct_prediction = formatted_prediction
            
            # Save the changes
            survey_results.save()
            print("Data entry successful.")
        except SurveyResults.DoesNotExist:
            print("No existing record found for the provided patient_id.")
        except IntegrityError as e:
            print(f"Data entry failed: {e}")

        # Create a directory to store the images if it doesn't exist
        image_directory = os.path.join(settings.MEDIA_ROOT, 'images')
        os.makedirs(image_directory, exist_ok=True)

        # Save the image to the directory
        image_path = os.path.join(image_directory, f"{patient_id}.jpg")
        cv2.imwrite(image_path, img)

        # Update the image_path field in the SurveyResults record
        survey_results.image_path = image_path
        survey_results.save()

        # Construct the URL with patient ID and prediction results
        url = reverse('image_report')
        patient_info_url = reverse('patient_info', args=[patient_id])
        url += f"?prediction={formatted_prediction}&patient_info_url={patient_info_url}&image_path={image_path}"

        # Redirect to the patient_info page with the prediction results and image path
        return redirect(url)

def display(request):
    # Retrieve all SurveyResults instances along with related PatientInfo instances
    survey_results_instances = SurveyResults.objects.select_related('patient_id').all()

    for survey_result in survey_results_instances:
        prediction = survey_result.ct_prediction
        max_label = None

        if prediction != None and prediction != 'None':
            prediction_data = ast.literal_eval(prediction)
            max_label = max(prediction_data, key=lambda x: x[1])[0]

        survey_result.max_label = max_label

    # Pass the data to the template for rendering
    return render(request, 'Display.html', {'survey_results_instances': survey_results_instances})

def patient_info(request, patient_id):
    # Get the PatientInfo instance
    patient_info = get_object_or_404(PatientInfo, patient_id=patient_id)

    # Get all SurveyResults instances for the given patient_id
    survey_results = SurveyResults.objects.filter(patient_id=patient_id)
    survey_results = survey_results.first()

    previousData = True

    prediction,prediction_data, max_label, max_prob = survey_results.ct_prediction, None, None, None

    if prediction != None and prediction != 'None':
        prediction_data = ast.literal_eval(prediction)
        max_label, max_prob = max(prediction_data, key=lambda x: x[1])

    # Pass the data to the template for rendering
    return render(request, 'PatientInfo.html', {'patient_info': patient_info, 'survey_results': survey_results, 'prediction': prediction_data, 'max_label': max_label, 'max_prob': max_prob, 'previousData':previousData})

def DoctorSignInPage(request):
    return render(request,'DoctorSignIn.html')

def DoctorSignIn(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            # Authentication successful, create session variable
            login(request, user)
            return redirect('Display')
        else:
            return render(request, 'DoctorSignIn.html', {'error_message': 'Invalid Username or Password'})
        
def SignOut(request):
    logout(request)
    return redirect('doctor_sign_in_page')

def Submitted(request):
    return render(request,'Submitted.html')

def ImageReport(request):
    prediction = request.GET.get('prediction', None)
    patient_info_url = request.GET.get('patient_info_url', None)
    image_path = request.GET.get('image_path', None)
    print(patient_info_url)

    prediction_data, max_label, max_prob = None, None, None

    if prediction != None and prediction != 'None':
        prediction_data = ast.literal_eval(prediction)
        max_label, max_prob = max(prediction_data, key=lambda x: x[1])

    # Pass the data to the template for rendering
    return render(request, 'ImageModelResult.html', {'prediction': prediction_data, 'max_label': max_label, 'max_prob': max_prob, 'patient_info_url': patient_info_url, 'image_path': image_path})