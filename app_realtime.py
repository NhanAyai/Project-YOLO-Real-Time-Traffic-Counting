from flask import Flask, render_template, request, Response, session
import plotly.express as px
from plotly.io import to_html
import pandas as pd
import json
from ultralytics import YOLO
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import yt_dlp
import cvzone
import cv2
import numpy as np
from pytube import YouTube
import time
from yolov8.tracker import* 
import plotly.graph_objects as go
from joblib import load

# from yolov8.video_processing import *

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Cần có secret_key để sử dụng session

# File paths
geojson_path = 'data/map.geojson'
# model_path = ['models/model_svm.joblib',
#               'models/model_DecisionTree.joblib',
#               'models/model_KNN.joblib',
#               'models/model_Linear_Regression.joblib',
#               'models/model_RandomForest.joblib',
#               'models/model_XGB.joblib'
#               ]

# Load GeoJSON data
def load_geojson():
    with open(geojson_path) as f:
        return json.load(f)


def get_livestream_url(youtube_url):
    ydl_opts = {
        'format': 'best',
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        best_stream_url = info_dict['url']
    return best_stream_url


youtube_urls = [
        "https://www.youtube.com/watch?v=6dp-bvQ7RWo",
        "https://www.youtube.com/watch?v=B0YjuKbVZ5w"
    ]

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), ['CarCount','BusCount','TruckCount', 'Total'])
#     ])

model_yolo = YOLO('yolov8/yolov8s.pt')
model_predict = load('C:/Users/Admin/Downloads/Project_realtime_2/Project/models/model_svm.joblib')
stream_url = ''
vehicle_counts = []
predicted_output = 0

# Biến để kiểm tra thời gian reset
last_reset_time = time.time()

def predictions(avg_count):
    cars, buses, trucks, counts = avg_count
    df = pd.DataFrame({
    'CarCount': int(cars),
    'BusCount': [int(buses)],
    'TruckCount': [int(trucks)], 
    'Total': [int(counts)]
    })

    # cars, buses, trucks, counts = 30, 0, 10, 40
    # df = pd.DataFrame({
    # 'CarCount': int(cars),
    # 'BusCount': [int(buses)],
    # 'TruckCount': [int(trucks)], 
    # 'Total': [int(counts)]
    # })
    return df


def generate_frames(stream_url):
    global vehicle_counts, predicted_output, model_yolo, model_predict
    global last_reset_time

    cap = cv2.VideoCapture(stream_url)
    tracker = Tracker()
    
    # Initialize counters and trackers
    
    cy1 = 200
    cy2 = 300
    offset = 6
    with open("yolov8/coco.txt", "r") as my_file:
        class_list = my_file.read().split("\n")

    while True:
        count = 0
        car_count = 0
        bus_count = 0
        truck_count = 0

        ret, frame = cap.read()  # Read a frame from the video
        if not ret:  # If no frame is read (end of video), break the loop
            break
        count += 1  # Increment frame count
        frame = cv2.resize(frame, (1020, 500))  # Resize the frame for consistent processing

        # Predict objects in the frame using YOLO model
        results = model_yolo.predict(frame)
        detections = results[0].boxes.data
        px = pd.DataFrame(detections).astype("float")  # Convert the prediction results into a pandas DataFrame

        # Initialize a list to store bounding boxes for each vehicle type
        cars, buses, trucks = [], [], []

        # Iterate over the detection results and categorize them into cars, buses, or trucks
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]
            if 'car' in c:
                cars.append([x1, y1, x2, y2])
            elif 'bus' in c:
                buses.append([x1, y1, x2, y2])
            elif 'truck' in c:
                trucks.append([x1, y1, x2, y2])

        # Update tracker for each vehicle type
        cars_boxes = tracker.update(cars)
        buses_boxes = tracker.update(buses)
        trucks_boxes = tracker.update(trucks)

        # cv2.line(frame, (1, cy1), (1018, cy1), (0, 255, 0), 2)
        # cv2.line(frame, (3, cy2), (1016, cy2), (0, 0, 255), 2)
        # Check each car, bus, and truck
        # global car_count, bus_count, truck_count
        for bbox in cars_boxes:
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            # if (cy > cy1 - offset) and (cy < cy1 + offset):
            #     car_count += 1

        for bbox in buses_boxes:
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            # if (cy > cy1 - offset) and (cy < cy1 + offset):
            #     bus_count += 1

        for bbox in trucks_boxes:
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            # if (cy > cy1 - offset) and (cy < cy1 + offset):
            #     truck_count += 1

        # Draw and annotate each vehicle
        for bbox in cars_boxes:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 2)
            cvzone.putTextRect(frame, f'Car', (bbox[0], bbox[1]), 1, 1)

        for bbox in buses_boxes:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 2)
            cvzone.putTextRect(frame, f'Bus', (bbox[0], bbox[1]), 1, 1)

        for bbox in trucks_boxes:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 2)
            cvzone.putTextRect(frame, f'Truck ', (bbox[0], bbox[1]), 1, 1)

        car_count = len(cars_boxes)
        bus_count = len(buses_boxes)
        truck_count = len(trucks_boxes)
        count = car_count + bus_count + truck_count

        vehicle_counts.append([car_count, bus_count, truck_count, count])
        current_time = time.time()
        if current_time - last_reset_time >= 5:
        # Reset lại danh sách và thời gian
            print(f"Resetting vehicle counts after {current_time - last_reset_time} seconds")
            avg_count = np.mean(vehicle_counts, axis=0)
            input_df = predictions(avg_count) 
            print(input_df)
            predicted_output = model_predict.predict(input_df)[0]
            print(predicted_output)
            # predicted_output = 1
            vehicle_counts = []  # Reset danh sách
            last_reset_time = current_time


        cvzone.putTextRect(frame, f'Car: {car_count}, Bus: {bus_count}, Truck: {truck_count}, Total: {count}', (400, 480), 2)
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame as part of a multipart HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        

@app.route('/', methods=['GET', 'POST'])
def index():
    global stream_url, predicted_output

    # Lấy video_index từ URL parameter (nếu có)
    video_index = request.args.get('video_index', default=0, type=int)

    # model_index = request.args.get('model_index', default=0, type=int)
     

    # model_names = ["SVM Model", "Decision Tree Model", "KNN Model", "Linear Regression Model", "Random Forest Model", "XGBoost Model"]
    # selected_model_name = model_names[model_index]

    # Cập nhật stream_url với video_index mới
    stream_url = get_livestream_url(youtube_urls[video_index])
    geojson_data = load_geojson()

    # Simulate choropleth data for the map
    data = {
        'name': ['Heavy', 'High', 'Low', 'Normal'],
        'value': [0, 1, 2, 3]
    }
    df = pd.DataFrame(data)

    # Plot the choropleth map
    if video_index == 0:
        fig = px.choropleth_mapbox(
            df,
            geojson=geojson_data,
            locations='name',
            color='name',
            mapbox_style='open-street-map',
            zoom=15,
            center={"lat": 35.6920, "lon": 139.7050},
            opacity=0.5,
            labels={'name': 'Traffic Situation'},
            width=750,
            height=560,
            color_discrete_map={
                "Heavy": "red",
                "High": "green",
                "Low": "blue",
                'Normal': 'yellow'
            },
        )

        # Add a line trace for Hai Bà Trưng street
        line_data = pd.DataFrame({
            'lat': [35.69315, 35.6920],
            'lon': [139.7030, 139.7075],
            'name': ['Start', 'End']
        })
        fig.add_trace(go.Scattermapbox(
            lat=line_data['lat'],
            lon=line_data['lon'],
            mode='lines+markers',
            marker=go.scattermapbox.Marker(size=8, color='red'),
            line=dict(width=4, color='blue'),
            name='Shinjuku'
        ))

        fig.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            legend=dict(
                x=0.98,
                y=0.98,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",  # Background color with some transparency
                bordercolor="black",
                borderwidth=1
            )
        )


    else:
        fig = px.choropleth_mapbox(
            df,
            geojson=geojson_data,
            locations='name',
            color='name',
            mapbox_style='open-street-map',
            zoom=15,
            center={"lat": 39.1921, "lon": -106.8165},
            opacity=0.5,
            labels={'name': 'Traffic Situation'},
            width=750,
            height=560,
            color_discrete_map={
                "Heavy": "red",
                "High": "green",
                "Low": "blue",
                'Normal': 'yellow'
            },
        )

        # Add line trace for Colorado
        line_data = pd.DataFrame({
            'lat': [39.1940, 39.1915],
            'lon': [-106.8178, -106.8190],
            'name': ['Start', 'End']
        })
        fig.add_trace(go.Scattermapbox(
            lat=line_data['lat'],
            lon=line_data['lon'],
            mode='lines+markers',
            marker=go.scattermapbox.Marker(size=8, color='red'),
            line=dict(width=4, color='blue'),
            name='Colorado'
        ))

        fig.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            legend=dict(
                x=0.98,
                y=0.98,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",  # Background color with some transparency
                bordercolor="black",
                borderwidth=1
            )
        )

    color_map = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow'}
    print(predicted_output)
    line_color = color_map.get(predicted_output, 'gray')


    fig.data[-1].line.color = line_color



    # Update the map with the new trace
    graph_html = fig.to_html(full_html=False)

    # Render the template with the prediction result
    return render_template('index.html', graph_html=graph_html, video_index=video_index)


@app.route('/video_feed')
def video_feed():
    
    return Response(generate_frames(stream_url), mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == '__main__':
    app.run(debug=True)
