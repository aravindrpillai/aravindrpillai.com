from stock.engine_y_finance import StockEngine
from property_reader import PropertyReader
from flask import jsonify
import datetime
import pytz


class StockHandler:
    
    def __init__(self, app, db):
        self.app = app
        self.db = db

    def analyse(self, request):
        try:
            data = request.get_json()
            if not data or 'ticker' not in data:
                return jsonify({"status": False, "message": "Ticker is required"}), 400
            
            days_back = data['days']
            if days_back == None or days_back == '':
                days_back = 3*365
            else:
                days_back = int(days_back) 

            code = PropertyReader.get_property("stock.access.code")
            pw = data['pw']
            if pw != code:
                toronto_time = datetime.datetime.now(pytz.timezone('America/Toronto'))
                time_number = int(toronto_time.strftime("%H%M"))
                time_number = time_number + 2020
                pw = int(pw)
                if(pw < (time_number-15) or pw > (time_number) ):
                    raise Exception("Incorrect Access Code!")

            ticker = data['ticker']  #"RELIANCE"
            predictor = StockEngine(ticker, days_back=days_back)
            resp = predictor.predict_today()
            return resp
        except Exception as e:
            return {"status": False, "message": f"Error: {str(e)}"}


    def predict(self, request):
        pass