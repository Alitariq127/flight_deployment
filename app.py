from flask import Flask, request, render_template
from flask_cors import cross_origin
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":

        # ── Departure Date & Time ──────────────────────────────────────────
        date_dep = request.form["Dep_Time"]
        dep_dt = pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M")
        Day_of_Journey    = int(dep_dt.day)
        Month_of_Journey  = int(dep_dt.month)
        Dep_hr            = int(dep_dt.hour)
        Dep_min           = int(dep_dt.minute)

        # ── Arrival Time ───────────────────────────────────────────────────
        date_arr = request.form["Arrival_Time"]
        arr_dt = pd.to_datetime(date_arr, format="%Y-%m-%dT%H:%M")
        Arrival_hr  = int(arr_dt.hour)
        Arrival_min = int(arr_dt.minute)

        # ── Duration (correctly calculated) ───────────────────────────────
        duration     = arr_dt - dep_dt
        duration_hr  = int(duration.seconds // 3600)
        duration_min = int((duration.seconds % 3600) // 60)

        # ── Total Stops ────────────────────────────────────────────────────
        Total_Stops = int(request.form["stops"])

        # ── Airline (one-hot) ──────────────────────────────────────────────
        airline_list = [
            'Air India', 'GoAir', 'IndiGo', 'Jet Airways',
            'Jet Airways Business', 'Multiple carriers',
            'Multiple carriers Premium economy', 'SpiceJet',
            'Trujet', 'Vistara', 'Vistara Premium economy'
        ]
        selected_airline = request.form['airline']
        airline_flags = {name: 1 if name == selected_airline else 0 for name in airline_list}

        # ── Source (one-hot) ───────────────────────────────────────────────
        source_list = ['Chennai', 'Delhi', 'Kolkata', 'Mumbai']
        selected_source = request.form["Source"]
        source_flags = {f"Source_{name}": 1 if name == selected_source else 0 for name in source_list}

        # ── Destination (one-hot) ──────────────────────────────────────────
        destination_list = ['Cochin', 'Delhi', 'Hyderabad', 'Kolkata', 'New Delhi']
        selected_destination = request.form["Destination"]
        dest_flags = {f"Destination_{name}": 1 if name == selected_destination else 0 for name in destination_list}

        # ── Predict (feature order matches training data exactly) ──────────
        prediction = model.predict([[
            airline_flags['Air India'],
            airline_flags['GoAir'],
            airline_flags['IndiGo'],
            airline_flags['Jet Airways'],
            airline_flags['Jet Airways Business'],
            airline_flags['Multiple carriers'],
            airline_flags['Multiple carriers Premium economy'],
            airline_flags['SpiceJet'],
            airline_flags['Trujet'],
            airline_flags['Vistara'],
            airline_flags['Vistara Premium economy'],
            source_flags['Source_Chennai'],
            source_flags['Source_Delhi'],
            source_flags['Source_Kolkata'],
            source_flags['Source_Mumbai'],
            dest_flags['Destination_Cochin'],
            dest_flags['Destination_Delhi'],
            dest_flags['Destination_Hyderabad'],
            dest_flags['Destination_Kolkata'],
            dest_flags['Destination_New Delhi'],
            Total_Stops,
            Day_of_Journey,
            Month_of_Journey,
            Dep_hr,
            Dep_min,
            Arrival_hr,
            Arrival_min,
            duration_hr,
            duration_min,
        ]])

        output = round(prediction[0], 2)
        return render_template('index.html',
                               prediction_text="Your Flight Price is Rs. {}".format(output))

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)