import os
import json
import threading
from datetime import date, datetime
from stock.intraday_predictor import IntradayPredictor 

class IntradayHandler:
 
    def __init__(self, app, db):
        self.app = app
        self.db = db

    def do_analysis(self, request):
        directory = os.path.join("static", "stock", "intraday", "reports")
        os.makedirs(directory, exist_ok=True)
        tomorrow_date = IntradayPredictor.get_next_weekday_from_today()
        filename = os.path.join(directory, f"{tomorrow_date}.json")
        temp_filename = os.path.join(directory, f"{tomorrow_date}.processing")
        if os.path.exists(filename):
            return {"status": True, "message": f"Prediction for {tomorrow_date} is already generated."}    
        elif os.path.exists(temp_filename):
            return {"status": True, "message": f"Prediction for {tomorrow_date} is in progress. Check after an hour."}    
        else:
            thread = threading.Thread(target=IntradayPredictor.start, daemon=True)
            thread.start()
            return {"status": True, "message": f"Predition for {tomorrow_date} has started. Check after an hour."}

    def pull_report(self, request):
        data = request.get_json()
        today_str = date.today().isoformat()
        self.generate_report_if_not_available = False
        if 'date' in data:
            prediction_date = data["date"]
            try:
                prediction_date = datetime.strptime(prediction_date, '%Y-%m-%d').date()
            except ValueError:
                raise ValueError("Date must be in YYYY-MM-DD format.")
        else:
            prediction_date = today_str

        directory = os.path.join("static", "stock", "intraday", "reports")
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, f"{prediction_date}.json")
        
        content = None
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                json_str = f.read()
                content = json.loads(json_str)
        
        return {"status": True, "report": content}
            
