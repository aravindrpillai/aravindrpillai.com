from property_reader import PropertyReader

class HandleMessages:
    def __init__(self, app, db):
        self.app = app
        self.db = db


    def handle(self, request):
        resp = []
        password = request.args.get('p')
        if(password != PropertyReader.get_property("app.password")):
            return {"error":"UnAuthorised"}
        else:
            if(request.args.get('action') == "delete"):
                self.handle_delete(request.args.get("id"))
                return 
            elif(request.args.get('action') == "read"):
                format = request.args.get('format')
                if format == "full":
                    return self.read_actual_data()
                else:
                    return self.read_actual_data_inline()
            else:
               return {"error":"UnAuthorised"} 

    def read_actual_data(self):
        resp = []
        data = self.db.read_messages()
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
    
    def read_actual_data_inline(self):
        resp = []
        data = self.db.read_messages()
        for row in data:
            resp.append("{} : {} -- {}".format(row[2],row[1],row[8]))
        return resp
    
    def handle_delete(self, id):
        if(id== None or id==""):
            raise Exception("No ID found")
        self.db.delete_messages(id)
        pass