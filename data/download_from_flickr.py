import sys, os
import pickle
import urllib.error
import urllib.request
import shutil
import random

if not os.path.exists("train"):
    os.mkdir("train")
if not os.path.exists("valid"):
    os.mkdir("valid")
if not os.path.exists("valid_ref"):
    os.mkdir("valid_ref")

with open('train_from_flickr.pkl', 'rb') as f:
    train_from_flickr = pickle.load(f)
with open('valid_from_flickr.pkl', 'rb') as f:
    valid_from_flickr = pickle.load(f)

for i, user_id in enumerate(train_from_flickr.keys()):
    dir_name = "train/"+user_id.replace("@", "_")+"/"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    for url_z in train_from_flickr[user_id]:
        print(i, user_id, url_z)
        try:
            with urllib.request.urlopen(url_z) as web_file:
                data = web_file.read()
                with open(dir_name+url_z.split("/")[-1], mode='wb') as local_file:
                    local_file.write(data)
        except KeyboardInterrupt:
            exit()
        except urllib.error.URLError as e:
            print(e)

for i, user_id in enumerate(valid_from_flickr.keys()):
    dir_name = "valid/"+user_id.replace("@", "_")+"/"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    for url_z in valid_from_flickr[user_id]:
        print(i, user_id, url_z)
        try:
            with urllib.request.urlopen(url_z) as web_file:
                data = web_file.read()
                with open(dir_name+url_z.split("/")[-1], mode='wb') as local_file:
                    local_file.write(data)
        except KeyboardInterrupt:
            exit()
        except urllib.error.URLError as e:
            print(e)

    dir_name_ref = "valid_ref/"+user_id.replace("@", "_")+"/"
    if not os.path.exists(dir_name_ref):
        os.mkdir(dir_name_ref)

    valid_ref_image_names = random.sample(os.listdir(dir_name), 20)
    for valid_ref_image_name in valid_ref_image_names:
        shutil.move(os.path.join(dir_name, valid_ref_image_name), dir_name_ref)
