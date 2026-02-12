import bson

def load_bson_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            bson_data = f.read()
        try:
            bson_dict = bson.loads(bson_data) # bson library
        except AttributeError as e:
            bson_dict = bson.decode(bson_data) # pymongo library

        return bson_dict
    except Exception as e:
        print(e)
        return None

def save_bson_dict(bson_dict, file_path):
    try:
        try:
            bson_data = bson.dumps(bson_dict)   # bson library
        except AttributeError:
            bson_data = bson.encode(bson_dict)  # pymongo library

        with open(file_path, 'wb') as f:
            f.write(bson_data)
    except Exception as e:
        print(e)
