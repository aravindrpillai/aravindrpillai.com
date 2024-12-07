import sqlite3
import traceback
from ip_util import get_ip_info
from flask import g
from sms import push_sms
from property_reader import PropertyReader

class AnonymousMessageDB:
    def __init__(self, app):
        self.app = app
        self.database_name = PropertyReader.get_property("sqllite.dbname")
        self.anonymous_table_name = PropertyReader.get_property("sqllite.anonymous_tablename")
        self.textbox_table_name = PropertyReader.get_property("sqllite.textbox_tablename")
        self.app.teardown_appcontext(self.close_db)
        
        # Ensure tables are created within the app context
        with app.app_context():
            self.create_tables()

    def get_db(self):
        if 'db' not in g:
            g.db = sqlite3.connect(self.database_name)
            g.db.row_factory = sqlite3.Row
        return g.db

    def close_db(self, e=None):
        db = g.pop('db', None)
        if db is not None:
            db.close()

    def create_tables(self):
        db = self.get_db()
        cursor = db.cursor()

        # Create the anonymous table
        anonymous_query = f'''
            CREATE TABLE IF NOT EXISTS {self.anonymous_table_name} (
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
        '''
        cursor.execute(anonymous_query)
        
        # Create the textbox table
        textbox_query = f'''
            CREATE TABLE IF NOT EXISTS {self.textbox_table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                code VARCHAR(20), 
                content TEXT
            )
        '''
        cursor.execute(textbox_query)

        db.commit()
        cursor.close()

    def insert_anonymous_message(self, message, ipaddress):
        db = self.get_db()
        cursor = db.cursor()
        location_data = get_ip_info(ipaddress)
        cursor.execute(
            f'INSERT INTO {self.anonymous_table_name} (message, ipaddress, latitude, longitude, area) VALUES (?, ?, ?, ?, ?)',
            (message, ipaddress, location_data['latitude'], location_data["longitude"], location_data["area"])
        )
        db.commit()
        cursor.close()

        try:
            pass
            # Uncomment to send SMS
            # sms_body = "{}\nIP:{} from {}\nhttps://www.google.com/maps?q={},{}".format(
            #     message, ipaddress, location_data["area"], location_data['latitude'], location_data["longitude"]
            # )
            # push_sms(sms_body)
        except Exception as e:
            traceback.print_exc()

    def read_anonymous_messages(self):
        db = self.get_db()
        cursor = db.cursor()
        cursor.execute(f'SELECT * FROM {self.anonymous_table_name}')
        rows = cursor.fetchall()
        cursor.close()
        return rows

    def delete_anonymous_messages(self, id):
        db = self.get_db()
        cursor = db.cursor()
        cursor.execute(f'DELETE FROM {self.anonymous_table_name} WHERE ID = ?', (id,))
        db.commit()
        cursor.close()

    def insert_textbox_data(self, code, content):
        db = self.get_db()
        cursor = db.cursor()

        cursor.execute(f"SELECT COUNT(*) FROM {self.textbox_table_name} WHERE code = ?", (code,))
        exists = cursor.fetchone()[0]

        if exists:
            cursor.execute(f"UPDATE {self.textbox_table_name} SET content = ? WHERE code = ?", (content, code))
        else:
            cursor.execute(f"INSERT INTO {self.textbox_table_name} (code, content) VALUES (?, ?)", (code, content))

        db.commit()
        cursor.close()

    def read_textbox_data(self, code):
        db = self.get_db()
        cursor = db.cursor()
        
        cursor.execute(f"SELECT content FROM {self.textbox_table_name} WHERE code = ? LIMIT 1", (code,))
        data = cursor.fetchone()

        if not data:
            cursor.execute(f"INSERT INTO {self.textbox_table_name} (code, content) VALUES (?, ?)", (code, None))
            db.commit()
            content = None
        else:
            content = data['content']

        cursor.close()
    
        return str(content) if content is not None else ""

    def close_connection(self):
        db = self.get_db()
        db.close()
