import configparser

class PropertyReader:
    @staticmethod
    def get_property(key):
        config = configparser.ConfigParser()
        prop_file = "constants.properties"
        config.read(prop_file)
        try:
            return config.get("main", key)
        except configparser.NoOptionError:
            return None