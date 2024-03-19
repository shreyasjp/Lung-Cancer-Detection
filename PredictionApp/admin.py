from django.contrib import admin

# Register your models here.
from .models import PatientInfo, SurveyResults

admin.site.register(PatientInfo)
admin.site.register(SurveyResults)