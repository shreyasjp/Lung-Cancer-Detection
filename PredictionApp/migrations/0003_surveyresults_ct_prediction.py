# Generated by Django 5.0 on 2024-03-10 06:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('PredictionApp', '0002_rename_sl_no_patientinfo_patient_id_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='surveyresults',
            name='ct_prediction',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]