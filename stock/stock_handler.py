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
            
            ticker = None if "ticker" not in data else data['ticker'].upper()
            exchange = None if "exchange" not in data else data['exchange'].upper()
            model_from_req = "v2" if 'model' not in data else data['model'].lower()
            days = (365*3) if "days" not in data else int(data['days'])
            pw = None if "pw" not in data else data['pw']
            
            #print(f'REQUEST :: {ticker} -- {exchange} -- {model_from_req} -- {days} -- {pw}')

            if ticker == None:
                raise Exception("Ticker not specified")
            if exchange == None:
                raise Exception("Exchange not specified")

            code = PropertyReader.get_property("stock.access.code")
            if pw != code:
                toronto_time = datetime.datetime.now(pytz.timezone('America/Toronto'))
                time_number = int(toronto_time.strftime("%H%M"))
                time_number = time_number + 2020
                pw = int(pw)
                if(pw < (time_number-15) or pw > (time_number) ):
                    raise Exception("Incorrect Access Code!")

            match model_from_req:
                case "xg":
                    from stock.engine_xg_boost import StockEngine as SE_V_XG
                    predictor = SE_V_XG(ticker=ticker, exchange=exchange, days_back=days)
                    model = "Version.XG.1.0 (xg)"
                case "v1":
                    from stock.engine_y_finance_v1 import StockEngine as SE_V1
                    predictor = SE_V1(ticker=ticker, exchange=exchange, days_back=days)
                    model = "Version.YFinance.1.0 (v1)"
                case "v2":
                    from stock.engine_y_finance_v2 import StockEngine as SE_V2
                    predictor = SE_V2(ticker=ticker, exchange=exchange, days_back=days)
                    model = "Version.YFinance.2.0 (v2)"
                case __:
                    raise Exception("Model Not Specified")
                
            resp = predictor.predict_today()
            resp["model"] = model

            return resp
        except Exception as e:
            return {"status": False, "message": "Error: "+str(e)}


    def predict(self, request):
        pass