import requests
from property_reader import PropertyReader

def get_ip_info(ip_number):
    
    key = PropertyReader.get_property("ip.key")
    latitude = 0.00
    longitude = 0.00
    city = "nil"
    postal_code = "nil"
        
    try:     
        API_URL = 'http://apiip.net/api/check?accessKey='+key+'&ip='+ip_number
        res = requests.get(API_URL)
        json_response = res.json()
        print(API_URL)
        print(json_response)
        latitude = json_response["latitude"]
        longitude = json_response["longitude"]
        city = json_response["city"]
        postal_code = json_response["postalCode"]
    
    except Exception as e:
        pass

    return {
        "latitude":latitude,
        "longitude" : longitude,
        "area" : city+", "+postal_code
    }
