import sqlite3
from ip_util import get_ip_info
from flask import g
from sms import push_sms
from property_reader import PropertyReader

class AnonymousMessageDB:
    def __init__(self, app):
        self.app = app
        self.database_name = PropertyReader.get_property("sqllite.dbname")
        self.table_name = PropertyReader.get_property("sqllite.tablename")
        self.app.teardown_appcontext(self.close_db)

    def get_db(self):
        if 'db' not in g:
            g.db = sqlite3.connect(self.database_name)
            g.db.row_factory = sqlite3.Row
        return g.db

    def close_db(self, e=None):
        db = g.pop('db', None)
        if db is not None:
            db.close()

    def create_table(self):
        db = self.get_db()
        cursor = db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS {} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT,
                create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                ipaddress TEXT,
                read BOOLEAN DEFAULT 0,
                hidden BOOLEAN DEFAULT 0,
                latitude TEXT,
                longitude TEXT,
                area TEXT
            )
        '''.format(self.table_name))
        db.commit()
        cursor.close()

    def insert_message(self, message, ipaddress):
        self.create_table()
        db = self.get_db()
        cursor = db.cursor()
        location_data = get_ip_info(ipaddress)
        cursor.execute('INSERT INTO {} (message, ipaddress, latitude, longitude, area) VALUES (?, ?, ?, ?, ?)'.format(self.table_name),
                       (message, ipaddress, location_data['latitude'], location_data["longitude"], location_data["area"]))
        db.commit()
        cursor.close()
        try:
            sms_body = "{}\nIP:{} from {}\nhttps://www.google.com/maps?q={},{}".format(message, ipaddress, location_data["area"], location_data['latitude'], location_data["longitude"])
            print(sms_body)
            push_sms(sms_body)
        except:
            print("Failed to send SMS")
            pass

    def read_messages(self):
        db = self.get_db()
        cursor = db.cursor()
        cursor.execute('SELECT * FROM {}'.format(self.table_name))
        rows = cursor.fetchall()
        cursor.close()
        return rows
    
    def delete_messages(self, id):
        db = self.get_db()
        cursor = db.cursor()
        cursor.execute('DELETE FROM {} WHERE ID={}'.format(self.table_name, id))
        db.commit()
        cursor.close()

    def close_connection(self):
        db = self.get_db()
        db.close()
