from flask import Flask, request
from flask_restful import Resource, Api
from sqllite import AnonymousMessageDB
from flask_cors import CORS
from write_message import WriteMessages
from handle_message import HandleMessages

app = Flask(__name__)
api = Api(app)
CORS(app)
db = AnonymousMessageDB(app)

class Anonimous(Resource):
    def get(self):
        return HandleMessages(app, db).handle(request)
        
    def post(self):
        return WriteMessages(app, db).write(request)
        
api.add_resource(Anonimous, '/')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)