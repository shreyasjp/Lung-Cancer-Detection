# Generated by Django 5.0 on 2024-03-09 09:07

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('PredictionApp', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='patientinfo',
            old_name='sl_no',
            new_name='patient_id',
        ),
        migrations.RemoveField(
            model_name='patientinfo',
            name='text_prediction',
        ),
        migrations.RemoveField(
            model_name='patientinfo',
            name='text_prediction_percentage',
        ),
        migrations.AddField(
            model_name='patientinfo',
            name='gender',
            field=models.CharField(default='Male', max_length=6),
            preserve_default=False,
        ),
        migrations.CreateModel(
            name='SurveyResults',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('smoking', models.IntegerField()),
                ('yellow_fingers', models.IntegerField()),
                ('anxiety', models.IntegerField()),
                ('peer_pressure', models.IntegerField()),
                ('chronic_disease', models.IntegerField()),
                ('fatigue', models.IntegerField()),
                ('allergy', models.IntegerField()),
                ('wheezing', models.IntegerField()),
                ('alcohol_consuming', models.IntegerField()),
                ('coughing', models.IntegerField()),
                ('shortness_of_breath', models.IntegerField()),
                ('swallowing_difficulty', models.IntegerField()),
                ('chest_pain', models.IntegerField()),
                ('text_prediction', models.BooleanField(default=False)),
                ('text_prediction_percentage', models.FloatField()),
                ('patient_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='PredictionApp.patientinfo')),
            ],
        ),
    ]