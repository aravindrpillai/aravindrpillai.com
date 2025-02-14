from flask import Flask, request
from flask_restful import Resource, Api
from sqllite import AnonymousMessageDB
from flask_cors import CORS
from write_anonymous import WriteAnonymous
from read_anonymous import ReadAnonymous
from text_app import TextBoxUtil
from stock.stock_handler import StockHandler
from stock.intraday_handler import IntradayHandler

app = Flask(__name__)
api = Api(app)
CORS(app)
db = AnonymousMessageDB(app)

class Anonimous(Resource): 
    def get(self):
        return ReadAnonymous(app, db).read(request)
        
    def post(self):
        return WriteAnonymous(app, db).write(request)
        
class TextBox(Resource):
    def get(self):
        return TextBoxUtil(app, db).read(request)
        
    def post(self):
        return TextBoxUtil(app, db).write(request)
        
class Stock(Resource):
    def post(self):
        return StockHandler(app, db).analyse(request)
    
class Intraday(Resource):
    def post(self, action):
        handler = IntradayHandler(app, db)
        action = str(action).strip().lower()
        if action == "report":
            return handler.pull_report(request)
        elif action == "analyse":
            return handler.do_analysis(request)
        else:
            return {"error": "Invalid action"}, 400

api.add_resource(TextBox, '/textbox')        
api.add_resource(Anonimous, '/anonymous')
api.add_resource(Stock, '/stock')
api.add_resource(Intraday, '/intraday/<string:action>')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)