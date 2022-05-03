# The data files will appear in the _raw_ folder (data/raw)
# The codebooks will appear in a separate _codebooks_ folder (data/codebooks)
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
    
    if filename.endswith((".tab",".zsav")):
        filepath = f'/raw/{filename}'
    elif filename.endswith((".pdf")):
        filepath = f'/codebooks/{filename}'
    
    file_id = file["dataFile"]["id"]
    print("File location {}, id {}".format(filepath, file_id))

    response = data_api.get_datafile(file_id)
    with open(filepath, "wb") as f:
        f.write(response.content)
