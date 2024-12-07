from property_reader import PropertyReader

class ReadAnonymous:
    def __init__(self, app, db):
        self.app = app
        self.db = db


    def read(self, request):
        password = request.args.get('p')
        if(password != PropertyReader.get_property("app.password")):
            return {"error":"UnAuthorised"}
    
        if(request.args.get('action') == "delete"):
            return self.handle_delete(request.args.get("id"))
        return self.read_data()
        
    def read_data(self):
        resp = []
        data = self.db.read_anonymous_messages()
        for row in data:
            resp.append({
                'ID': row[0], 
                "Message": row[1], 
                "Time": row[2], 
                "IP": row[3], 
                "Read": row[4], 
                "Hidden": row[5], 
                "Latitude": row[6], 
                "Longitude": row[7], 
                "Area": row[8],
                "Map" : 'https://www.google.com/maps?q={},{}'.format(row[6], row[7])
            })
        return resp
    
    def handle_delete(self, id):
        try:
            if(id== None or id==""):
                raise Exception("No ID found")
            self.db.delete_anonymous_messages(id)
            return {"status":True, "message":None} 
        except Exception as e:
            return {"status":False, "message":f"Error: {str(e)}"} 