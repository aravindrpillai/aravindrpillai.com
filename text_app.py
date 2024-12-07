from flask import jsonify

class TextBoxUtil:
    
    def __init__(self, app, db):
        self.app = app
        self.db = db

    def write(self, request):
        try:
            data = request.get_json()
            if not data or 'content' not in data:
                return jsonify({"status": False, "message": "Content is required"}), 400

            code = data['code']
            if not code:
                return jsonify({"status": False, "message": "Code is required"}), 400

            content = data['content']
            print("Content received:", content)

            self.db.insert_textbox_data(code, content)  # Store the content directly
            return {'status': True, "message": None}
        except Exception as e:
            return {"status": False, "message": f"Error: {str(e)}"}
    
  
    def read(self, request):
        code = request.args.get('code')
        if not code or code == None or code == "":
            return {"status":True, "content":None, "message" : "Code missing"} 
    
        content = self.db.read_textbox_data(code)
        return {"status":True, "content":content, "message" : None} 
