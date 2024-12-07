class WriteAnonymous:
    def __init__(self, app, db):
        self.app = app
        self.db = db


    def write(self, request):
        try:
            data = request.get_json()
            message = data['message']
            if(message == None or message == ""):
                raise Exception("No Message")
            client_ip = request.remote_addr
            self.db.insert_anonymous_message(message, client_ip)
            return {'status': True, message : None}
        except Exception as e:
            return {"status":False, "message":f"Error: {str(e)}"} 

        