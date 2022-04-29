# The data files and codebooks will appear in the same folder as this file 
from pyDataverse.api import NativeApi, DataAccessApi

API_TOKEN = '' # insert API token here
BASE_URL = 'https://data.aussda.at'
DOI = "doi:10.11587/I7QIYJ"
api = NativeApi(BASE_URL, API_TOKEN)
dataset = api.get_dataset(DOI)
data_api = DataAccessApi(BASE_URL, API_TOKEN)

files_list = dataset.json()['data']['latestVersion']['files']

for file in files_list:
    filename = file["dataFile"]["filename"]
    file_id = file["dataFile"]["id"]
    print("File name {}, id {}".format(filename, file_id))

    response = data_api.get_datafile(file_id)
    with open(filename, "wb") as f:
        f.write(response.content)
