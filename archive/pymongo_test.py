#%%
from pymongo import MongoClient


client = MongoClient("mongodb://172.26.132.124:27017")
db = client.countries
collection = db.country

post = {'name': 'Liechtenstein'}
collection.insert_one(post)
#post_id = collection.insert_one(post)
#post_id = collection.insert_one(post).inserted_id

# %%
