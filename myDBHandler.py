from pymongo import MongoClient

# Handles the communication with the Mongo DB


class MyDBHandler:

    # Constructor - Initiates the connection with Mongo/DB using Mongo client
    def __init__(self, my_db):
        self.myClient = MongoClient("mongodb+srv://marman:5marman@cluster0-cin8u.mongodb.net/" + my_db + "?retryWrites=true&w=majority")
        self.myDB = self.myClient[my_db]
        print("Database Connection established successfully!")

    # Inserts data/a tweet in collection
    def insert_data(self, collection, inserted_data):
        result = self.myDB[collection].insert_one(inserted_data)
        return result

    # Deletes all data from collection
    def delete_all_data(self, collection):
        self.myDB[collection].delete_many({})

    # Fetch data from collection
    def get_data(self, collection, item):
        cursor = self.myDB[collection].find({})
        result = []
        for document in cursor:
            result.append(document[item])
        return result

    # Fetch data from all collections
    def get_all_data(self, prefix, item):
        collections = self.myDB.collection_names()
        result = []
        for collection in collections:
            if collection.startswith(prefix):
                result.extend(self.get_data(collection, item))
        return result

    # Fetch data from collection
    def get_categories_data(self, collection, item):
        cursor = self.myDB[collection].find({})
        result = []
        for document in cursor:
            smart_category = collection.split("_")[1]
            result.append(smart_category)
        return result

    # Fetch data from all collections
    def get_all_categories_data(self, prefix, item):
        collections = self.myDB.collection_names()
        result = []
        for collection in collections:
            if collection.startswith(prefix):
                result.extend(self.get_categories_data(collection, item))
        return result