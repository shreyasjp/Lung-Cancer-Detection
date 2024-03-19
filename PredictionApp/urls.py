from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path('process_survey',views.process_survey,name='ProcessForm'),
    path('process_image',views.process_image,name='ProcessImage'),
    path('display',views.display, name='Display'),
    path('patient_info',views.patient_info, name='patient_info'),
    path('patient_info/<int:patient_id>/', views.patient_info, name='patient_info'),
    path('doctor_sign_in_page',views.DoctorSignInPage, name='doctor_sign_in_page'),
    path('doctor_sign_in',views.DoctorSignIn, name='doctor_sign_in'),
    path('sign_out',views.SignOut, name='sign_out'),
    path('success',views.Submitted, name='success'),
    path('image_report',views.ImageReport, name='image_report'),
    path('survey',views.survey, name='survey'),    
]