
# coding: utf-8

# In[26]:

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk 
import sqlalchemy
import seaborn as sns
import urllib2
from bs4 import BeautifulSoup
import requests
import zipfile
from StringIO import StringIO
import gzip

get_ipython().magic(u'matplotlib inline')


# In[28]:

def GetFile(filename):
    baseURL = "http://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/"

    response = urllib2.urlopen(baseURL + filename)
    compressedFile = StringIO()
    compressedFile.write(response.read())
    #
    # Set the file's current position to the beginning
    # of the file so that gzip.GzipFile can read
    # its contents from the top.
    #
    compressedFile.seek(0)

    decompressedFile = gzip.GzipFile(fileobj=compressedFile, mode='rb')

    return decompressedFile


# In[30]:

def GetFileList():
    weatherDataUrl = "http://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
    soup = BeautifulSoup(requests.get(weatherDataUrl).text, "lxml")
    fileList = []
    for a in soup.find_all('a'):
            if "StormEvents" in a["href"]:
                fileList.append(a["href"])
    return fileList


# In[32]:

def GetFile_Convert_Append(filename,df):
    csvFile = GetFile(filename)
    csv = pd.read_csv(csvFile)
    csv.columns = map(str.lower,csv.columns)
    #csv.to_sql("details",con = engine, if_exists = "replace", chunksize=500)
    df = df.append(csv)
    return df


# In[50]:

def Collect():
    engine = sqlalchemy.create_engine('postgresql://alexandertam@localhost/')
    
    fileTypes = ["details","locations","fatalities"]
    years = range(2015,2016)
    years = map(str,years)
    detailsDF = pd.DataFrame()
    locationsDF = pd.DataFrame()
    fatalitiesDF = pd.DataFrame()
    
    for filename in GetFileList():
        if any(year in filename for year in years):
            if("details" in filename):
                detailsDF = GetFile_Convert_Append(filename,detailsDF)
                continue
            if("fatalities" in filename):
                fatalitiesDF = GetFile_Convert_Append(filename,fatalitiesDF)
                continue
            if("locations" in filename):
                locationsDF = GetFile_Convert_Append(filename,locationsDF)
                continue
            
    TornadoesDF = detailsDF[detailsDF["event_type"]=="Tornado"]
    
    fatalitiesDF.to_sql("fatalities",con = engine, if_exists = "replace")
    
    AlabamaDF = detailsDF[detailsDF["state"] ==  "ALABAMA"].copy()
    AlabamaDF = AlabamaDF[AlabamaDF["wfo"] != "TAE"]
    AlabamaDF = AlabamaDF[["wfo","episode_id","event_id","event_type","begin_date_time","end_date_time","begin_lat","begin_lon"]]
    AlabamaDF = AlabamaDF.dropna()
    AlabamaDF.to_sql("alabama",con = engine, if_exists = "replace")
    locationsDF.to_sql("locations",con = engine, if_exists = "replace")


# In[ ]:



