####load libraries####
import pandas as pd
import numpy as np
####

###load data####
prop24 = pd.read_csv("DSI/Regression Twist Challenge/Property24_HousePrice.csv")
sahome = pd.read_csv("DSI/Regression Twist Challenge/SAHometraders_HousePrice.csv")

prop24.head()
sahome.head()

####format prop24 data####
prop24.info()

#remove m^2 in floorSize
prop24["floorSize"] = prop24["floorSize"].str[:-3]
prop24["floorSize"] = pd.to_numeric(prop24["floorSize"], errors='coerce')

#extract propery type for description
propertyType_p24 = []
for i in prop24.index:
    propertyType_p24.append(prop24["type"][i].split()[-1])
prop24["propertyType"] = propertyType_p24

#remove rows without a price
prop24 = prop24[prop24["price"].notna()]

#remove rows with more than two missing values
prop24_dropped = prop24.dropna(thresh = prop24.shape[1]-1)

####format sahome data####
sahome.head()
sahome.info()

#remove m^2 in floorSize
sahome["floorSize"] = sahome["floorSize"].str[:-3]
sahome["floorSize"] = pd.to_numeric(sahome["floorSize"], errors='coerce')

#extract propery type for description
propertyType_sahome = []
for i in sahome.index:
    propertyType_sahome.append(sahome["type"][i].split()[-1])
sahome["propertyType"] = propertyType_sahome

#convert "Flat" to "Apartment" to match prop24
sahome["propertyType"] = sahome["propertyType"].str.replace("Flat", "Apartment")

#format price
sahome["price"] = sahome["price"].str.replace(u'\xa0', u' ').str.lstrip("R").str.replace(" ", "")
sahome["price"] = pd.to_numeric(sahome["price"], errors='coerce')

#remove rows without a price
sahome = sahome[sahome["price"].notna()]

#remove rows with more than two missing values
sahome_dropped = sahome.dropna(thresh = sahome.shape[1]-1)

####Merge Prop24 and SAhome dataframe####
house_prices_merge = pd.concat([prop24_dropped, sahome_dropped], ignore_index = True)
house_prices_merge = house_prices_merge.drop(["type"], axis = 1)

#format numerical data that is not yet as correct type
house_prices_merge["bedroom"] = pd.to_numeric(house_prices_merge["bedroom"], errors='coerce')
house_prices_merge["bathroom"] = pd.to_numeric(house_prices_merge["bathroom"], errors='coerce')

##remove completely duplicated entries in the merged dataframe
house_prices_final = house_prices_merge.drop_duplicates()

#export the final merged dataframe as CSV
house_prices_final.to_csv("DSI/Regression Twist Challenge/HousePrice_Final.csv", index=False, header=True)