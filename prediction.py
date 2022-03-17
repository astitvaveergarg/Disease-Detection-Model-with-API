import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB


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

#Training
dataset=pd.read_csv(r"dataset\Training.csv")

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

#Testing
tr=pd.read_csv(r"dataset\Testing.csv")
tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30, 'Hypothyroidism':31,'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,'Impetigo':40}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)


def NB_prediction():
    gnb = GaussianNB()
    gnb=gnb.fit(X,np.ravel(y))
 
    symptoms = [S1,S2,S3,S4,S5]
    for k in range(0,len(l1)):
        for z in symptoms:
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
        t = ("PREDICTED DISEASE: ", disease[a])
    else:
        t = ("Not Found")

    print(t)
    print("#####################################################################")
    print("Doctor Recomendation: ")

    if a==0 or a==1 or a==4 or a==16 or a==37 or a==40 or a==39:
        print("Doctor Name: Dr. Ramkumar Verma \nDesignation: Dermatalogist \nQualification: MBBS, MD \nRating: 4.8 \nConsultation Fee: Rs.800 per Sitting" )

    elif a==2 or a==3 or a==5 or a==8 or a==19 or a==20 or a==21 or a==22 or a==23 or a==24:
        print("Doctor Name: Dr. Vasudha Sharma \nDesignation: Gastroenterologist \nQualification: MBBS, MD \nRating: 4.6 \nConsultation Fee: Rs.500 per Sitting" )

    elif a==12 or a==38:
        print("Doctor Name: Dr. Sharda Singh \nDesignation: Gynacologist \nQualification: MBBS, MD \nRating: 5.0 \nConsultation Fee: Rs.1500 per Sitting" )

    elif a==34 or a==35:
        print("Doctor Name: Dr. Tanishq Kashyap \nDesignation: Orthologist \nQualification: MBBS, MD \nRating: 4.2 \nConsultation Fee: Rs.500 per Sitting" )
    
    elif a==9 or a==27 or a==25:
        print("Doctor Name: Dr. Sanskar Gupta \nDesignation: Pulmonologist \nQualification: MBBS, MD \nRating: 4.7 \nConsultation Fee: Rs1800 per Sitting" )

    elif a==7 or a==14 or a==15 or a==17 or a==18 or a==26:
        print("Doctor Name: Dr. Adrika Kakoty \nDesignation: Physician \nQualification: MBBS, MD \nRating: 4.6 \nConsultation Fee: Rs500 per Sitting" )

    elif a==29 or a==30:
        print("Doctor Name: Dr. Astitva Garg \nDesignation: Cardiologist \nQualification: MBBS, MD \nRating: 5.0 \nConsultation Fee: Rs2000 per Sitting" )
    
    elif a==6 or a==31 or a==32:
        print("Doctor Name: Dr. Samaresh Samanta \nDesignation: Endocrinologist \nQualification: MBBS, MD \nRating: 5.0 \nConsultation Fee: Rs1500 per Sitting" )

    elif a==10 or a==11 or a==13 or a==28 or a==36:
        print("Doctor Name: Dr. Anishka Kesaria \nDesignation: Neurologist \nQualification: MBBS, MD \nRating: 4.9 \nConsultation Fee: Rs1500 per Sitting" )

    print("######################################################################")
    print("######################################################################")

symptomsgg=input("Enter Symptoms: ")
symptom = symptomsgg.split(",")

S1 = symptom[0]
S2 = symptom[1]
S3 = symptom[2]
S4 = symptom[3]
S5 = symptom[4]

NB_prediction()