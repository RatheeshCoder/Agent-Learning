from app.database.connection import db
user_collection = db["users"]

def create_user(user_data: dict):
    result = user_collection.insert_one(user_data)
    print(f"User created with ID: {result}")
    return str(result.inserted_id)

def get_all_users():
    users = user_collection.find({}, {"_id": 0})  
    return list(users)
