####load libraries
import requests
import pandas as pd
import numpy as np
from scrapy import Selector
import time
####

####standard set up
#create base locator path
root_path = 'span.p24_content'

# set up the feature paths
price_locator = 'span.p24_price::attr(content)'
location_locator = 'span.p24_location::text'
bedroom_locator = 'span.p24_featureDetails[title = Bedrooms] > span::text'
bathroom_locator = 'span.p24_featureDetails[title = Bathrooms] > span::text'
garage_locator = 'span.p24_featureDetails[title = "Parking Spaces"] > span::text'
floorSize_locator = 'span.p24_size > span::text'
type_locator = 'span.p24_title::text'

# collect feature paths into list
locator_list = [price_locator, location_locator, bedroom_locator, bathroom_locator, garage_locator, floorSize_locator, type_locator]

#initialise variables to hold features
price = ""; location = ""; bedroom = ""; bathroom = ""; garage = ""; floorSize = ""; type = ""

# collect variables into a list
feature_list = [price, location, bedroom, bathroom, garage, floorSize, type]

#zip feature paths and feature variables together
feature_package = list(zip(feature_list, locator_list))

#create DataFrame to hold final results
house_price = pd.DataFrame(columns = ["price", "location", "bedroom", "bathroom", "garage", "floorSize", "type"])
####

###set up urls
#page 1 url
urls = ["https://www.property24.com/for-sale/cape-town/western-cape/432?PropertyCategory=House%2cApartmentOrFlat%2cTownhouse"]

#manually generate the urls for pages 2 to the end
urls_comp = ["https://www.property24.com/for-sale/cape-town/western-cape/432/p{0}?PropertyCategory=House%2cApartmentOrFlat%2cTownhouse".format(i) for i in range(2, 398)]

#combine page 1 url and urls of subsequent search pages
urls.extend(urls_comp)

#loop over urls
for url in urls:
    #get html document from url
    html= requests.get(url).content

    #instantiate selector object
    sel = Selector(text = html)

    #generator base selector
    root_locator = sel.css(root_path)

    ###---future feature: automatically find next page and add url to for-each loop---###
    #get url of next page
    # url2_path = '//a[@data-pagenumber = 2]/@href' ##add formating to make number dynamic; add loop to continue until end
    # url2 = sel.xpath(url2_path).extract()[0]
    # urls.append(url2)

    #loop over the selector object (each advert panel)
    for selector in root_locator:
        temp_list = []

        #iteratate over the feature paths and extract the values into the feature variable
        for feature, locator in feature_package:
            try:
                #chain the base locator with the specific feature path and extract the value to the feature variable
                feature = selector.css(locator).extract_first().strip()
                temp_list.append(feature)

            #if an error occurs becuase the value is missing for this entry, insert NaN
            except:
                feature = np.nan
                temp_list.append(feature)

        #append the features of particular house as temp list as a row in the dataframe
        house_price = house_price.append(pd.Series(temp_list, index=house_price.columns), ignore_index=True)

    #delay next round of code
    time.sleep(10)

#drop duplicate entries in dataframe
df_dropped = house_price.drop_duplicates()

#export dataframe to csv file
df_dropped.to_csv("DSI/Regression Twist Challenge/Property24_HousePrice.csv", index=False, header=True)