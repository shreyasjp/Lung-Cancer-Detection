from django.db import models

# Create your models here.
class PatientInfo(models.Model):
    patient_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    gender = models.CharField(max_length=6)

class SurveyResults(models.Model):
    patient_id = models.ForeignKey(PatientInfo, on_delete=models.CASCADE)
    smoking = models.IntegerField()
    yellow_fingers = models.IntegerField()
    anxiety = models.IntegerField()
    peer_pressure = models.IntegerField()
    chronic_disease = models.IntegerField()
    fatigue = models.IntegerField()
    allergy = models.IntegerField()
    wheezing = models.IntegerField()
    alcohol_consuming = models.IntegerField()
    coughing = models.IntegerField()
    shortness_of_breath = models.IntegerField()
    swallowing_difficulty = models.IntegerField()
    chest_pain = models.IntegerField()
    text_prediction = models.BooleanField(default=False)
    text_prediction_percentage = models.FloatField()
    ct_prediction = models.CharField(max_length = 255, blank=True, null=True)
    image_path = models.URLField(max_length=255, blank=True, null=True) 