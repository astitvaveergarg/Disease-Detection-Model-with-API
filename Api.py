from flask import Flask, jsonify
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)

@app.route('/')
def main():
    return "api is calling"

@app.route('/symptoms=<string:symptomsgg>')
def Predicted(symptomsgg):
    global l1
    global l2
    l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
    'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
    'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
    'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
    'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
    'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
    'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
    'yellow_crust_ooze']

    disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
    'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
    ' Migraine','Cervical spondylosis',
    'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
    'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
    'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
    'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
    'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
    'Impetigo']

    l2=[]
    for x in range(0,len(l1)):
        l2.append(0)

    dataset=pd.read_csv(r"./Dataset/Training.csv")

    dataset.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
    'Migraine':11,'Cervical spondylosis':12,
    'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
    'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
    'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
    'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
    '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
    'Impetigo':40}},inplace=True)


    X= dataset[l1]

    y = dataset[["prognosis"]]
    np.ravel(y)

    tr=pd.read_csv(r"./Dataset/Testing.csv")
    tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
    'Migraine':11,'Cervical spondylosis':12,'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30, 'Hypothyroidism':31,'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,'Impetigo':40}},inplace=True)

    X_test= tr[l1]
    y_test = tr[["prognosis"]]
    np.ravel(y_test)

    symptom = symptomsgg.split("&")
    print(symptom)
    
    gnb = GaussianNB()
    gnb=gnb.fit(X,np.ravel(y))
    
    for k in range(0,len(l1)):
        for z in symptom:
            if(z==l1[k]):
                l2[k]=1

    input_test = [l2]
    predict = gnb.predict(input_test)
    predicted=predict[0]

    m='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            m='yes'
            break

    if (m=='yes'):
        t = disease[a]
    else:
        t = "Not Found"

    if a==0 or a==1 or a==4 or a==16 or a==37 or a==40 or a==39:
        result = {
            "Deasease": t,
            "Doctor Name":"Dr. Ramkumar Verma",
            "Designation": "Dermatalogist", 
            "Qualification":"MBBS, MD",
            "Rating": 4.8,
            "Consultation Fee": 800,
            "Tests": "Blood Test, Allergy Test, Prick Test"}

    elif a==2 or a==3 or a==5 or a==8 or a==19 or a==20 or a==21 or a==22 or a==23 or a==24:
        result = {
            "Deasease": t,
            "Doctor Name": "Dr. Vasudha Sharma",
            "Designation": "Gastroenterologist",
            "Qualification": "MBBS, MD",
            "Rating": 4.6,
            "Consultation Fee": 500,
            "Tests": "Endoscopy, Colonoscopy, Sigmoidoscopy"}

    elif a==12 or a==38:
        result = {
            "Deasease": t,
            "Doctor Name" : "Dr. Sharda Singh", 
            "Designation": "Gynacologist", 
            "Qualification": "MBBS, MD", 
            "Rating": 5.0, 
            "Consultation Fee": 1500,
            "Tests": "Ultrasound, STD Test, Biopsy"}

    elif a==34 or a==35:
        result = {
            "Deasease": t,
            "Doctor Name" : "Dr. Tanishq Kashyap", 
            "Designation": "Orthologist", 
            "Qualification": "MBBS, MD", 
            "Rating": 4.2, 
            "Consultation Fee": 500,
            "Tests": "Arthrography, X-Ray, CT Scan"}
        
    elif a==9 or a==27 or a==25:
        result = {
            "Deasease": t,
            "Doctor Name" : "Dr. Sanskar Gupta", 
            "Designation": "Pulmonologist", 
            "Qualification": "MBBS, MD", 
            "Rating": 4.7, 
            "Consultation Fee": 1800,
            "Tests": "Spirometry, Plethysmography, CT scan"}

    elif a==7 or a==14 or a==15 or a==17 or a==18 or a==26:
        result = {
            "Deasease": t,
            "Doctor Name" : "Dr. Adrika Kakoty", 
            "Designation": "Physician", 
            "Qualification": "MBBS, MD", 
            "Rating": 4.6, 
            "Consultation Fee": 500,
            "Test": "Blood Test, Maleria Test Dengue Test"}

    elif a==29 or a==30:
        result = {
            "Deasease": t,
            "Doctor Name" : "Dr. Astitva Garg", 
            "Designation": "Cardiologist", 
            "Qualification": "MBBS, MD", 
            "Rating": 5.0, 
            "Consultation Fee": 2000,
            "Test": "Endoscopy, ECG, Colour Doppler"}
        
    elif a==6 or a==31 or a==32:
        result = {
            "Deasease": t,
            "Doctor Name" : "Dr. Samaresh Samanta", 
            "Designation": "Endocrinologist", 
            "Qualification": "MBBS, MD", 
            "Rating": 5.0, 
            "Consultation Fee": 1500,
            "Test": "Complete Blood Count, Thyroid, ACTH Level"}
    elif a==10 or a==11 or a==13 or a==28 or a==36:
        result = {
            "Deasease": t,
            "Doctor Name" : "Dr. Anishka Kesaria", 
            "Designation": "Neurologist", 
            "Qualification": "MBBS, MD", 
            "Rating": 4.9, 
            "Consultation Fee": 1500,
            "Test": "MRI, CT Scan, PET"}

    return jsonify(result)


@app.route('/symptom')
def Symptom():
    l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
    'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
    'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
    'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
    'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
    'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
    'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
    'yellow_crust_ooze']
    return jsonify(l1)

if __name__ == '__main__':
    app.run(threaded=False,debug=True)
