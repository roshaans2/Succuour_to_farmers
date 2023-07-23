from flask import Flask,render_template, url_for ,flash , redirect, Markup
from flask import request
from flask import send_from_directory
from requests_html import HTMLSession
import requests

import folium
import requests
import pandas as pd


from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

import pickle


from keras.models import load_model

import os


import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from IPython.display import HTML


from tensorflow.keras.utils import load_img, img_to_array


import numpy as np

from utils.disease import disease_dic


import mysql.connector


from utils.fertilizer import fertilizer_dict


from market_stat import Market





app=Flask(__name__,template_folder='template')


mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="roshaan@2002",
    database="sf"
)

mycursor = mydb.cursor()


model = load_model("Trained.h5")


rf_model_path = 'RandomForest.pkl'
rf_model = pickle.load(open(rf_model_path, 'rb'))


df = pd.read_csv("Cities_with_lat_long.csv")
df = df.head(10)
API_KEY = "4898bf479091d9ff49c3001f88e91659"
temperatures = []
humidities = []
descriptions = []

m = folium.Map(location=[20.5937, 78.9629], zoom_start=4)

def getData():
    for index, row in df.iterrows():
        city_name = row['Name of City']
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            temperature = data['main']['temp']
            humidity = data['main']['humidity']
            desc = data["weather"][0]["description"]
            temperatures.append(temperature)
            humidities.append(humidity)
            descriptions.append(desc)
        else:
            print(f"Error fetching weather data for {city_name}")

    # Add the weather data to the DataFrame
    df['Temperature'] = temperatures
    df['Humidity'] = humidities
    df['Description'] = descriptions




def getMap():
   
    for index, row in df.iterrows():
        name = row['Name of City']
        lat = row['lat']
        lon = row['lng']
        temp = row['Temperature']
        hum = row['Humidity']
        desc = row['Description']
        popup_text = f"{name}<br>Temperature: {temp} C<br>Humidity: {hum}%<br>Description: {desc}"
        folium.Marker(location=[lat, lon], popup=popup_text).add_to(m)
        
    m.save(r"C:\Users\ROSHAAN\Desktop\Rotaract\Template\output.html")
    
    
getData()
getMap()

class_names = dict()
class_names['peach'] = ['Peach___Bacterial_spot', 'Peach___healthy']
class_names['apple'] = ['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy']
class_names['cherry'] = ['Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy']
class_names['corn'] = ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight',
'Corn_(maize)___healthy']
class_names['grape'] = ['Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy']
class_names['pepper'] = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy']
class_names['potato'] = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
class_names['strawberry'] = ['Strawberry___Leaf_scorch', 'Strawberry___healthy']
class_names['tomato'] = ['Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot','Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato__Tomato_mosaic_virus','Tomato_healthy']


def sendMessage(number,message):
    import requests


    account_sid = 'AC792037010ba00073b6e956ece5a86b9d'
    auth_token = '046fbc8d8ba8a527b88a748fe0b7e566'

 
    from_number = '+13203723588'

    to_number = number

 
    url = f'https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json'


    params = {
        'From': from_number,
        'To': to_number,
        'Body': message
    }

  
    r = requests.post(url, auth=(account_sid, auth_token), data=params)

  
    print(r.text)


def predict(model, img,crop):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    class_name = class_names[crop]
    predicted_class = class_name[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence
    

@app.route('/weatherMap')
def weather_map():
    return render_template('output.html')



@app.route('/weatherForecast', methods=['POST'])
def weather_forecast():
    if request.method == "POST":
        city_name = request.form['city_name']
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
      
        desc = data["weather"][0]["description"]
        print(humidity)
        print(temperature)
        print(desc)
        
        url1 = "http://api.openweathermap.org/data/2.5/forecast"
        params = {
            "q": city_name,
            "appid": API_KEY,
            "units": "metric"
        }
        response1 = requests.get(url1, params=params)

        # Check if the request was successful
        if response1.status_code == 200:
            # Convert the response JSON string into a Python dictionary
            data = response1.json()
            table_data  = []
            # Extract the forecast data
            forecast = data["list"]

            # Print the forecast for each 3-hour interval
            for item in forecast:
                date_time = item["dt_txt"]
                temp = item["main"]["temp"]
                weather_desc = item["weather"][0]["description"]
                #print(f"{date_time}: {temp}°C, {weather_desc}")
                dic = dict()
                dic['timestamp'] = date_time
                dic['temperature'] = temp
                dic['desc'] = weather_desc
                
                table_data.append(dic)
              

        else:
            print(f"Error: {response1.status_code}")
        
        return render_template("weather_data.html",city_name=city_name,temp=temperature,humidity=humidity,desc = desc,table_data=table_data)


    
@app.route('/weatherVisualize',methods=['POST'])
def wether_visualize():
    if request.method == 'POST':
        city_name = request.form['city_name']
        url1 = "http://api.openweathermap.org/data/2.5/forecast"
        params = {
        "q": city_name,
         "appid": API_KEY,
         "units": "metric"
         }
        response1 = requests.get(url1, params=params)
         # Check if the request was successful
        if response1.status_code == 200:
                # Convert the response JSON string into a Python dictionary
                data = response1.json()
                table_data  = []
                # Extract the forecast data
                forecast = data["list"]

                # Print the forecast for each 3-hour interval
                for item in forecast:
                    date_time = item["dt_txt"]
                    temp = item["main"]["temp"]
                    weather_desc = item["weather"][0]["description"]
                    #print(f"{date_time}: {temp}°C, {weather_desc}")
                    dic = dict()
                    dic['timestamp'] = str(date_time)
                    dic['temperature'] = temp
                    dic['desc'] = weather_desc

                    table_data.append(dic)


        else:
             print(f"Error: {response1.status_code}")

        dv = pd.DataFrame(table_data)
        dv.to_csv(r'C:\Users\ROSHAAN\Desktop\Rotaract\template\dataSet.csv')
    
    return render_template('visualize.html')


@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/weather')
def weather():
    return render_template('weather.html')

@app.route('/disease')
def disease():
    return render_template('disease.html');


@app.route('/disease-predict', methods=['POST'])
def disease_prediction():

    if request.method == 'POST':
        crop = request.form['crop']
        file = request.files.get('file')
        filename = file.filename
        print(filename)
        image_path_jpg = os.path.join(r'C:\Users\ROSHAAN\Desktop\Rotaract\static\user uploaded', filename)
        print(image_path_jpg)
        file.save(image_path_jpg)
        img = tf.io.read_file(image_path_jpg)
        img.get_shape().as_list()  
        img = tf.image.decode_jpeg(img)
        img.get_shape().as_list()  
        img_resized = tf.image.resize(img, [256, 256])
        img_resized.get_shape().as_list()  
        
        model = load_model(crop+".h5")

        prediction = predict(model,img_resized,crop)
        print(prediction)
        
        confidence = prediction[1]

        prediction = Markup(str(disease_dic[prediction[0]]))

    return render_template('disease-result.html', prediction=prediction,confidence = confidence)
#         except:
#             pass
#     return render_template('disease.html', title=title)



@app.route("/Crop_recommend")
def Crop_recommend():
    return render_template("crop_recommend.html")



@app.route('/crop_prediction', methods=['POST'])
def crop_prediction():
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['potassium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = rf_model.predict(data)
        final_prediction = my_prediction[0]
    return render_template('crop-result.html', prediction=final_prediction, pred='img/crop/'+final_prediction+'.jpg')


@app.route("/Donate")
def Donate():
    mycursor.execute("SELECT * FROM dnte")
    data = mycursor.fetchall()
    print(data)
    return render_template("donate_new.html",data=data)


@app.route("/donate_forms")
def Donate_form():
   
    return render_template("donate_forms.html")


@app.route("/donate_forms_response", methods=['POST'])
def Donate_form_response():
    title = str(request.form['title'])
    content = str(request.form['content'])
    gpay = str(request.form['gpay'])
    
    message = 'We recieved your request and will get back to you shortly' + '\n' + 'உங்கள் கோரிக்கையை நாங்கள் பெற்றுள்ளோம், விரைவில் உங்களைத் தொடர்புகொள்வோம்' + '\n' + 'हमें आपका अनुरोध प्राप्त हो गया है और हम शीघ्र ही आपसे संपर्क करेंगे' + '\n' + 'మేము మీ అభ్యర్థనను స్వీకరించాము మరియు త్వరలో మిమ్మల్ని సంప్రదిస్తాము' + '\n' + 'നിങ്ങളുടെ അഭ്യർത്ഥന ഞങ്ങൾക്ക് ലഭിച്ചു, ഉടൻ തന്നെ നിങ്ങളെ ബന്ധപ്പെടും' + '\n' + 'ನಿಮ್ಮ ವಿನಂತಿಯನ್ನು ನಾವು ಸ್ವೀಕರಿಸಿದ್ದೇವೆ ಮತ್ತು ಶೀಘ್ರದಲ್ಲೇ ನಿಮ್ಮನ್ನು ಸಂಪರ್ಕಿಸುತ್ತೇವೆ'
    
    sendMessage('+91'+ gpay,message)
    
    sql = "INSERT INTO dnte (title,content,gpay) VALUES (%s, %s, %s)"
    val = (title, content, gpay)
    mycursor.execute(sql, val)
    mydb.commit()
    
    return redirect('/Donate')



@app.route("/Complaint")
def Complaint():
    return render_template("complaint.html")



@app.route("/complaintResponse", methods=['POST'])
def Complaint_response():
    
    if request.method == 'POST':
        name = str(request.form['name'])
        email = str(request.form['email'])
        pno = str(request.form['phonenumber'])
        complaint = str(request.form['complaint'])
        
        message = 'We have recieved your complaint. We will respond to it as soon as possible' + '\n' + 'உங்கள் புகாரை நாங்கள் பெற்றுள்ளோம். கூடிய விரைவில் அதற்கு பதிலளிப்போம்' + '\n' + 'हमें आपकी शिकायत मिली है। हम इसका जल्द से जल्द जवाब देंगे' + '\n' + 'మేము మీ ఫిర్యాదును స్వీకరించాము. వీలైనంత త్వరగా దీనిపై స్పందిస్తాం' + '\n' + 'നിങ്ങളുടെ പരാതി ഞങ്ങൾക്ക് ലഭിച്ചു. ഞങ്ങൾ അതിനോട് എത്രയും വേഗം പ്രതികരിക്കും' + '\n' + 'ನಿಮ್ಮ ದೂರನ್ನು ನಾವು ಸ್ವೀಕರಿಸಿದ್ದೇವೆ. ಅದಕ್ಕೆ ಆದಷ್ಟು ಬೇಗ ಸ್ಪಂದಿಸುತ್ತೇವೆ'
        
        sendMessage('+91' + pno,message)
    
    sql = "INSERT INTO complaint (name,email,pno,complaint) VALUES (%s, %s, %s, %s)"
    val = (name,email,pno,complaint)
    mycursor.execute(sql, val)
    mydb.commit()
    
    return render_template("complaint.html")


@app.route("/Fertilizer_recommend")
def Fertilizer_recommend():
    return render_template("fert_form.html")


@app.route('/fertilizer-predict', methods=['POST'])
def fertilizer_recommend():

    crop_name = str(request.form['cropname'])
    N_filled = int(request.form['nitrogen'])
    P_filled = int(request.form['phosphorous'])
    K_filled = int(request.form['potassium'])

    df = pd.read_csv('Crop_NPK.csv')

    N_desired = df[df['Crop'] == crop_name]['N'].iloc[0]
    P_desired = df[df['Crop'] == crop_name]['P'].iloc[0]
    K_desired = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = N_desired- N_filled
    p = P_desired - P_filled
    k = K_desired - K_filled

    if n < 0:
        key1 = "NHigh"
    elif n > 0:
        key1 = "Nlow"
    else:
        key1 = "NNo"

    if p < 0:
        key2 = "PHigh"
    elif p > 0:
        key2 = "Plow"
    else:
        key2 = "PNo"

    if k < 0:
        key3 = "KHigh"
    elif k > 0:
        key3 = "Klow"
    else:
        key3 = "KNo"

    abs_n = abs(n)
    abs_p = abs(p)
    abs_k = abs(k)

    response1 = Markup(str(fertilizer_dict[key1]))
    response2 = Markup(str(fertilizer_dict[key2]))
    response3 = Markup(str(fertilizer_dict[key3]))
    return render_template('Fertilizer-Result.html', recommendation1=response1,recommendation2=response2, recommendation3=response3,diff_n = abs_n, diff_p = abs_p, diff_k = abs_k)

@app.route('/taskSchedule')
def taskSchedule():
    return render_template('task.html')


@app.route("/PesticideRecommendation")
def pesticide():
    return render_template("PesticideRecommendation.html")

@app.route("/dp", methods=['GET', 'POST'])
def dp():
    if request.method == 'POST':
        file = request.files['image']  # fetch input
        print(file)
        filename = file.filename
        file_path = os.path.join(r'static\user uploaded', filename)
        file.save(file_path)
        try:
            test_image = load_img(file_path, target_size=(150, 150))
            test_image = img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = model.predict(test_image)
            print(result)
            pred_arr = []
            for i in result:
                pred_arr.extend(i)
            
            pred = pred_arr.index(1.0)
            
        except:
            pred =  'x'
        
        if pred == 'x':
            return render_template('unaptfile.html')
        if pred == 0:
            pest_identified = 'aphids'
        elif pred == 1:
            pest_identified = 'armyworm'
        elif pred == 2:
            pest_identified = 'beetle'
        elif pred == 3:
            pest_identified = 'bollworm'
        elif pred == 4:
            pest_identified = 'earthworm'
        elif pred == 5:
            pest_identified = 'grasshopper'
        elif pred == 6:
            pest_identified = 'mites'
        elif pred == 7:
            pest_identified = 'mosquito'
        elif pred == 8:
            pest_identified = 'sawfly'
        elif pred == 9:
            pest_identified = 'stem borer'

    return render_template(pest_identified + ".html",pred=pest_identified)
    


from flask import jsonify

@app.route('/data')
def get_data():
    data = {
        'name': 'John',
        'age': 30,
        'city': 'New York'
    }
    return jsonify(data)


    
    
@app.route('/Scheme')
def scheme():
    return render_template('scheme.html')

@app.route('/weatherVisualize')
def weather_visualize():
    return render_template('weatherVisualize.html')

@app.route('/')
def login():
    return render_template('login.html')


@app.route('/IncorrectLogin')
def incorrectLogin():
    return render_template('Incorrectlogin.html')

@app.route("/LoginResponse", methods=['POST'])
def login_response():
    
    if request.method == 'POST':
        mno = str(request.form['Mobile_Number'])
        ps = str(request.form['Password'])
        email = str(request.form['Email'])
        
        sql1 = "SELECT * FROM users WHERE mno = %s"
        val1 = []
        val1.append(mno)
        mycursor.execute(sql1,val1)
        data1 = mycursor.fetchall()
        print(data1)
        
        if(not data1):
            print('yes')
            sql = "INSERT INTO users (mno,ps,email) VALUES (%s, %s, %s)"
            val = (mno, ps, email)
            mycursor.execute(sql, val)
            mydb.commit()
            return redirect('/home')
        
        else:
            for item in data1:
                if(item[0] != mno or item[1] != ps or item[2] != email):
                    return redirect('/IncorrectLogin')
#                     return redirect('/Login')
           

            
        
 

        
#         mycursor.execute("SELECT * FROM users")
#         data = mycursor.fetchall()
#         for item in data:
#             if(item[0] != mno and item[1] != ps and item[2] != email):
#                 sql = "INSERT INTO users (mno,ps,email) VALUES (%s, %s, %s)"
#                 val = (mno, ps, email)
#                 mycursor.execute(sql, val)
#                 mydb.commit()
#                 return redirect('/home')

#             elif(item[0] == mno and item[1]!= ps):
#                 return redirect('/Login')
        
        
    
        
    
    return redirect('/home')

@app.route('/market',methods=['POST','GET'])
def market():

    model = Market()
    states,crops = model.State_Crop()
    if request.method == 'POST':
        state = request.form['state']
        crop = request.form['crop']
        lt = model.predict_data(state,crop)

        return render_template('market.html',result=lt,result_len =len(lt),display=True,states=states,crops=crops)

    return render_template('market.html',states=states,crops=crops) 


@app.route('/cropMonitoring')
def crop_montoring():
    model =YOLO("yolov8s.pt")
    cap = cv2.VideoCapture(0)
    results=model.predict(source="0", show=True)

    while True:
        ret, frame = cap.read()
        cv2.imshow('frame',frame)
        key = cv2.waitKey(1)
        if key  == ord('q'):
            break

    cap.release()      
    cv2.destroyAllWindows()

    print(results)



if __name__ == "__main__":
    app.run(debug=True)
    
    

    
        