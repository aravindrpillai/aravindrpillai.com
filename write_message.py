import traceback

class WriteMessages:
    def __init__(self, app, db):
        self.app = app
        self.db = db


    def write(self, request):
        status = True
        try:
            data = request.get_json()
            message = data['message']
            if(message == None or message == ""):
                raise Exception("No Message")
            
            client_ip = request.remote_addr
            self.db.insert_message(message, client_ip)
        except Exception as e:
            print('Failed to handle message : {}\n{}'.format(e, traceback.format_exc()))
            status= False

        return {'status': status}
    


