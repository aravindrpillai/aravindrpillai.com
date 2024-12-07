from flask import Flask, request
from flask_restful import Resource, Api
from sqllite import AnonymousMessageDB
from write_anonymous import WriteAnonymous
from read_anonymous import ReadAnonymous
from text_app import TextBoxUtil

app = Flask(__name__)
api = Api(app)
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

api.add_resource(TextBox, '/textbox')        
api.add_resource(Anonimous, '/anonymous')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=False)