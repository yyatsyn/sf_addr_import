import geopandas as gpd
import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from titlecase import titlecase
import xml.etree.ElementTree as ET
import requests
import os
from time import sleep
# ... on top of that, the system utility "osmfilter" is needed

# set the path to working directory
#path = '/home/yury/Desktop/sf_addr/'
place_name = "San Francisco, California, USA"
abspath = os.path.abspath(__file__)
path = os.path.dirname(abspath)
os.chdir(path)

#=======================================================================
# Read and pre-process the 3 required layers: addr points, parcels, buildings
#=======================================================================
#******* OSM Streets ********
osm_str = ox.graph_from_place(place_name, \
                                network_type = 'all_private', \
                                simplify = False, \
                                retain_all = True, \
                                buffer_dist = 100)
gdf_nodes, osm_str = ox.save_load.graph_to_gdfs(osm_str)
osm_str = osm_str.loc[~(osm_str['name'].isna())]


#******* Address points ********

# Read the shp-files with addresses and parcels
addr = gpd.read_file(path + 'geo_export_257813a1-7658-4a10-8c51-9174722bb849.shp')

# Replace <NONE> in street names and types with ""
addr.loc[addr['street_nam'].isna(), 'street_nam'] = ""
addr.loc[addr['street_typ'].isna(), 'street_typ'] = ""

# Generate a proper "full_street_name" field (as "High Street")
# load the table with abbreviations
abbrevs = pd.read_csv(path + 'maine_911_roads_abbreviations.csv')

# convert UPPER case to Title Case (as in shp-files roads_911)
abbrevs['full_suffix'] = abbrevs['full_suffix'].str.title()
#abbrevs['abbrev_suffix'] = abbrevs['abbrev_suffix'].str.title()

# merge the shp-file with the abbreviation crosswalk
addr = addr.merge(abbrevs, \
                            left_on = 'street_typ', \
                            right_on = 'abbrev_suffix', how = 'left')
# replace NaN in "full_suffix" with ""
addr.loc[addr["full_suffix"].isna(),'full_suffix']=""

# compose a full street name
# Convert to title only non-numeric names (to prevent "22Nd" instead of "22nd")
# use regular expressions for "^[0-9]ST$|ND$|RD$|TH$"
addr['numeric_name'] = addr['street_nam'].str.contains('^[0-9]+(ST$|ND$|RD$|TH)$', regex = True)
addr['numeric_name_zero'] = addr['street_nam'].str.contains('^0.', regex = True)
addr['street_nam_title'] = ""
# numeric names that start with "0"
addr.loc[addr['numeric_name_zero'], 'street_nam_title'] = \
    addr.loc[addr['numeric_name_zero'], 'street_nam'].str[1:].str.lower()
# numeric names that start with non-0
addr.loc[(~addr['numeric_name_zero']) & (addr['numeric_name']), 'street_nam_title'] = \
    addr.loc[(~addr['numeric_name_zero']) & (addr['numeric_name']), 'street_nam'].str.lower()
# non-numeric names
#addr.loc[(~addr['numeric_name']), 'street_nam_title'] = \
#    addr.loc[(~addr['numeric_name']), 'street_nam'].str.title()
addr.loc[(~addr['numeric_name']), 'street_nam_title'] = \
    addr.loc[(~addr['numeric_name'])].apply(lambda X: titlecase(X["street_nam"]), axis = 1)

# combine "Title Name" with "Street" suffix
addr["full_street_name"] = addr["street_nam_title"].str.cat(addr["full_suffix"],\
                                                        sep = " ")

# trim excessive whitespace (for the cases when suffix is empty)
addr["full_street_name"] = addr["full_street_name"].str.strip()

# CORRECT the erroneous names (the list is created manually after checking with OSM names)
# see the "STEP 0" below
addr_osm_replace = {'Bay Shore Boulevard': 'Bayshore Boulevard',\
                    'El Camino Del Mar': 'El Camino del Mar',\
                    'Florentine Avenue': 'Florentine Street',\
                    'La Playa': 'La Playa Street',\
                    'Ofarrell Street': "O'Farrell Street",\
                    'Oreilly Avenue': "O'Reilly Avenue",\
                    'Oshaughnessy Boulevard': "O'Shaughnessy Boulevard",\
                    'Bannan Place': "Bannam Place",\
                    "Emerald Cove Way": "Emerald Cove Terrace",\
                    "Heritage Lane": "Heritage Avenue",\
                    'Stanyan Boulevard': 'Stanyan Street'}

for k, v in addr_osm_replace.items():
    addr["full_street_name"] = addr["full_street_name"].str.replace(k, v)


# drop temp variables
addr.drop(['full_suffix', 'abbrev_suffix', 'street_nam', 'street_typ',\
    'numeric_name_zero', 'numeric_name', 'street_nam_title'], axis = 1, inplace = True)

# convert housenumbers and ID to integers
addr['address_nu'] = addr['address_nu'].astype(int)
addr['eas_baseid'] = addr['eas_baseid'].astype(int)


#******* Parcels ********

prcl = gpd.read_file(path + 'geo_export_a547c2ff-1c46-457c-8360-d5d8a56c1d50.shp')

# remove inactive/retired parcels
prcl = prcl.loc[(prcl['active'] == 'T') & \
    (prcl['mapblklot'] == prcl['blklot']) & \
    (prcl['date_rec_d'].isna()) & \
    (prcl['date_map_2'].isna()) & \
    (prcl['date_map_d'].isna())]
#prcl.to_file(path + 'sf_unique_parcels.shp')

# keep unique parcels
G = prcl["geometry"].apply(lambda geom: geom.wkb)
prcl = prcl.loc[G.drop_duplicates().index]
prcl = prcl.reset_index()
#prcl.to_file(path + 'sf_unique_parcels2.shp')

# Exclude parcels that intersect with other parcels
# intersect parcel layer with itself to find the overlapping parcels
# (it looks like some overlaps are still there)
# drop parcels that have non-trivial overlaps (consider by area)
prcl_check = prcl[['geometry', 'blklot']]
prcl_check['area'] = prcl_check['geometry'].area
prcl_overlap = gpd.overlay(prcl_check, prcl_check, how = 'intersection')
prcl_overlap = prcl_overlap.loc[~(prcl_overlap['blklot_1'] == prcl_overlap['blklot_2'])]
prcl_overlap['area'] = prcl_overlap['geometry'].area

# keep records of parcels that produce overlay of more than 20m^2
drop_prcl = prcl_overlap.loc[prcl_overlap['area'] > 20, 'blklot_1'].unique()
prcl = prcl[~(prcl["blklot"].isin(drop_prcl))]


#******* Buildings ********

# download and read the OSM-buildings from San-Francisco
# (using the osmnx library)
blds = ox.footprints.footprints_from_place(place_name)
# generate full_id, osm_type (way,relation) base on index and nodes
blds['osm_type'] = ''
blds.loc[blds['nodes'].isna(), 'osm_type'] = 'r' # no nodes => relation
blds.loc[blds['nodes'].notna(), 'osm_type'] = 'w' #

# check is there are buildings with a single node (that should be treated as a point)
# ... there are no such buildings
#blds_nodes = blds.loc[blds['osm_type']=='w', 'nodes']
#blds_nodes.apply(len).unique()==1

# generate a "full_id"-variable
blds['osm_id'] = blds.index.astype(str)
blds['full_id'] = blds['osm_type'].str.cat(blds['osm_id'], sep='')

#******* All 3 needed layers ********
# keep only needed tags
osm_str = osm_str[['geometry', 'osmid', 'name', 'highway']]
blds = blds[['geometry', 'addr:street', 'addr:housenumber', 'addr:unit', 'full_id']]
addr = addr[['geometry', 'address_nu', 'address__2','full_street_name', 'eas_baseid']]
prcl = prcl[['geometry', 'index', 'blklot']]

# make sure that CRS's in all 3 layers are the same
sf_epsg = 7131 # units -- meters
# init_epsg = 4326
osm_str = osm_str.to_crs(epsg = sf_epsg)
addr = addr.to_crs(epsg = sf_epsg)
blds = blds.to_crs(epsg = sf_epsg)
prcl = prcl.to_crs(epsg = sf_epsg)

# compute areas of parcels and buildings
prcl['prcl_area'] = prcl['geometry'].area
blds['blds_area'] = blds['geometry'].area

# compute the number of points in buildings external contours
n_vertices=[]
for i, row in blds.iterrows():
    # for multipolygons
    if row.geometry.geom_type == 'MultiPolygon':
        geoms_area = [np.abs(x.area) for x in row.geometry.geoms]
        largest_geom_index = geoms_area.index(np.max(geoms_area))
        n = len(row.geometry.geoms[largest_geom_index].exterior.coords) - 1 # the first and last points are the same
    # for Polygons
    elif row.geometry.geom_type == 'Polygon':
        n = len(row.geometry.exterior.coords) - 1
    else:
        break
    
    n_vertices.append(n)

blds["n_vertices"] = n_vertices


#=======================================================================
# Matching step 0: match addr to OSM streets to check for possible mistakes
#=======================================================================
### A FUNCTION that uses rtree spacial index to match two layers --
### ... -- osm_str and addr(points) -- to find K nearby streets and from
### them check k closest and return the name that is the most similar

def closest_name_of_nearby_streets(osm_str, addr, str_name='name', addr_name='name', K=20, k=10):
    # Both input layers are supposed to have "geometry" and "name" fields
    # The function returns the addr_layer augmented with two fields:
    # 1) the closest osm street name match
    # 2) the difflib.SequenceMatcher distance between addr- and OSM-names
    import difflib
    
    # Generate empty fields for storing the matched values
    addr['osm_name'] = ''
    addr['osm_name_match'] = 0.0
    
    # Create spacial index
    str_idx = osm_str["geometry"].sindex
    
    # Create a list of nearest streets for each address point
    addr = addr.reset_index(drop=True)
    addr_nearest_str = [list(str_idx.nearest((addr.loc[i,"geometry"].x, addr.loc[i,"geometry"].y), K, objects='raw')) for i in range(len(addr))]
    
    # Compute the actual distance from the address point to pre-selected K streets 
    for i in range(len(addr)):
        #print("Processing address " + str(i) + " out of " + str(len(addr)))
        # find actual distance from the address point "i" to K nearest streets (aacording to R-tree spatial index)
        dist = []
        for j in range(K):
            dist.append(addr.loc[i,"geometry"].distance(osm_str["geometry"][addr_nearest_str[i][j]]))
        
        # sort actual distances (from smallest to largest)
        srtd = dist[:]
        srtd.sort()
        # select k nearest streets
        c = [addr_nearest_str[i][l] for l in [dist.index(srtd[i]) for i in range(k)]]
        nearest_streets_names = set(osm_str.loc[c, str_name])
        nearest_streets_names.discard(None)
        
        cur_street = addr.loc[i, addr_name]
        #nearest_streets = [remove_simple_mismatches(x, to_remove, abbrevs) for x in nearest_streets_names]
        nearest_streets = nearest_streets_names
        
        # Compute the match score
        match_score = [difflib.SequenceMatcher(None, cur_street, x).ratio() for x in nearest_streets]
        addr['osm_name'][i] = list(nearest_streets)[match_score.index(max(match_score))]
        addr['osm_name_match'][i] = max(match_score)
    
    return addr

addr_matched = closest_name_of_nearby_streets(osm_str, \
                                                addr, \
                                                str_name = 'name', \
                                                addr_name = 'full_street_name', \
                                                K = 50, \
                                                k = 50)

addr_unmatched_to_osm = addr_matched.loc[addr_matched['osm_name_match'] < 1.0]
addr_unmatched_to_osm.to_csv(path + 'unmatched_streets.csv',\
    columns = ['address_nu', 'full_street_name', 'eas_baseid', 'osm_name', 'osm_name_match'])


#=======================================================================
# Matching step 1: match addr points to parcels 
#=======================================================================

# Match addr-points to parcels
addr_prcl = gpd.sjoin(addr, prcl, how = "inner", op = "intersects")
# check number of parcels to which each point is matched,
# ...keep only those addresses that are matched  to 1 parcel
addr_prcl['n_prcls'] = addr_prcl.groupby('eas_baseid')['blklot'].transform('nunique')
addr_prcl = addr_prcl.loc[addr_prcl['n_prcls'] == 1]
unmatched_addr = addr.loc[~(addr['eas_baseid'].isin(addr_prcl['eas_baseid']))]

# Loop: create a .01, .02, ..., 3 m buffers around addr-points to match
# ... those points that are close parcels, but not inside them 
# Loop over radii of buffers 
br = 0.01

while (br < 3.01):
    #print("Buffer size: " + str(br) + " m")
    
    # replace points with buffer polygons of radius=br
    unmatched_addr["geometry"] = unmatched_addr.geometry.buffer(br)
    
    # sjoin addresses with buffers
    addr_prcl_buff = gpd.sjoin(unmatched_addr, prcl, how = "inner", op = "intersects")
    
    # select addr points that are matched to exactly 1 parcel
    addr_prcl_buff['n_prcls'] = addr_prcl_buff.groupby('eas_baseid')['blklot'].transform('nunique')
    addr_prcl_buff = addr_prcl_buff.loc[addr_prcl_buff['n_prcls'] == 1]
    
    # append the resulting dataframe
    addr_prcl = addr_prcl.append(addr_prcl_buff)
    
    # subset unmatched address points
    unmatched_addr = addr.loc[~(addr['eas_baseid'].isin(addr_prcl['eas_baseid']))]
    
    # increase radius of buffers
    br += 0.01

# Check that the street names for all "addr" within each "prcl" are the same
addr_prcl['n_str_names'] = addr_prcl.groupby('blklot')['full_street_name'].transform('nunique')

# Exclude addresses that, when matched to parcels, ...
# ...result in multiple street names within a parcel
addr_prcl = addr_prcl.loc[addr_prcl['n_str_names'] == 1]
unmatched_addr = addr.loc[~(addr['eas_baseid'].isin(addr_prcl['eas_baseid']))]

# Drop temp variables
addr_prcl.drop(['n_str_names', 'n_prcls', 'index', 'index_right'], axis = 1, inplace = True)

# Create an indicator of whether addr:unit is needed:
# consider addr:unit as important if without it addr:street & addr:housenumber will be assigned to >1 blklot's
addr_prcl.loc[addr_prcl["address__2"].isna(),'address__2']=""
addr_prcl['blk_w_unit'] = addr_prcl.groupby(['address_nu', 'address__2', 'full_street_name'])['blklot'].transform('nunique')
addr_prcl['blk_wo_unit'] = addr_prcl.groupby(['address_nu', 'full_street_name'])['blklot'].transform('nunique')
addr_prcl['keep_unit'] = (addr_prcl['blk_wo_unit'] > addr_prcl['blk_w_unit']) * 1

# Combine addr__2 (where keep_unit==1) into ;-separated lists IF all housenumbers within the blklot are the same
# (otherwise) addresses should be imported as points... (and, hence, added to unmatched_addr)
addr_prcl['all_same_housenumber'] = addr_prcl.groupby(['blklot'])['address_nu'].transform('nunique')
# keep addresses that do not need addr:unit OR those that need addr:unit for which all housenumbers within blklots are the same
addr_prcl = addr_prcl.loc[(~(addr_prcl['keep_unit'] == 1)) | \
    ((addr_prcl['keep_unit'] == 1) & (addr_prcl['all_same_housenumber'] == 1))]
unmatched_addr = addr.loc[~(addr['eas_baseid'].isin(addr_prcl['eas_baseid']))]
# Drop temp variables
addr_prcl.drop(['blk_w_unit', 'blk_wo_unit', 'all_same_housenumber'], axis = 1, inplace = True)

# Combine address__2 into ;-separated string
addr_prcl = addr_prcl.set_index(['blklot'])
#addr_prcl.loc[addr_prcl["address__2"]=="",'address__2'] = None
addr_prcl['address__2'] = addr_prcl['address__2'].astype(str)
addr_prcl['joint_address_2'] = addr_prcl.groupby('blklot')['address__2'].apply(set) # to keep only unique numbers (to avoid "160;160;162" if the existing and imported numbers are the same)
addr_prcl['joint_address_2'] = addr_prcl['joint_address_2'].apply(lambda x: ";".join(sorted(x)))
# remove units from parcels on which they are not needed
addr_prcl.loc[~(addr_prcl['keep_unit'] == 1), 'joint_address_2'] = ''

# Combine address_nu into ;-separated string
addr_prcl['address_nu'] = addr_prcl['address_nu'].astype(str)
addr_prcl['joint_address_nu'] = addr_prcl.groupby('blklot')['address_nu'].apply(set) # to keep only unique numbers (to avoid "160;160;162" if the existing and imported numbers are the same)
#addr_prcl['joint_address_nu'] = addr_prcl['joint_address_nu'].apply(lambda x: ";".join(sorted(x)))
addr_prcl['joint_address_nu'] = addr_prcl['joint_address_nu'].apply(lambda x: ";".join(sorted(x)))

# Create an indicator for whether any address matched to a blklot ...
# ...is among the "addr_unmatched_to_osm" addresses (those that were not matched to any nearby OSM street)
addr_prcl['no_osm_str'] = addr_prcl['eas_baseid'].isin(addr_unmatched_to_osm['eas_baseid'])
addr_prcl['any_no_osm_str'] = addr_prcl.groupby('blklot')['no_osm_str'].transform(any)
# Keep only addresses that have matching OSM streets in proximity
addr_prcl = addr_prcl.loc[~(addr_prcl['any_no_osm_str'])]
unmatched_addr = addr.loc[~(addr['eas_baseid'].isin(addr_prcl['eas_baseid']))]

# Keep only required variables
addr_prcl_shrt = addr_prcl[['full_street_name', 'joint_address_nu', 'joint_address_2']].drop_duplicates()
addr_prcl_shrt.reset_index(level=0, inplace=True)


#=======================================================================
# Matching step 2: match buildings to parcels, ...
# ... identify buildings that should get addr:* tags for import 
#=======================================================================
# mark smallest buildings (shelters, barns, gander houses) to exclude them from matching
data = np.array(np.log(blds['blds_area']))
density = scipy.stats.gaussian_kde(data)
blds['area_pct'] = [density.integrate_box_1d(0, x)
    for x in np.log(blds['blds_area'])]

# sjoin buildings and parcels
blds_prcl = gpd.overlay(blds, prcl)

# find area of the buildings-parcels intersections
blds_prcl['joint_area'] = blds_prcl['geometry'].area
# merge with parcels-address frame ("bpa" -- for buildings-parcels-addresses)
bpa = blds_prcl.merge(addr_prcl_shrt, on = 'blklot', how = 'left')

# B <-> P (1-to-1)
# create the "Building <-> Parcel" concordance (bp)
# match buildings to parcels that contain contains >X% (say, >98%) of the building's area
bp = bpa[['full_id', 'blklot', 'blds_area', 'joint_area', 'area_pct', 'n_vertices']]
bp['tot_joint_area'] = bp.groupby(['full_id'])['joint_area'].transform(np.sum)
bp['max_joint_area'] = bp.groupby(['full_id'])['joint_area'].transform(np.max)
bp['tot_joint_area_share'] = bp['tot_joint_area'] / bp['blds_area']
bp['max_joint_area_share'] = bp['max_joint_area'] / bp['tot_joint_area']
# match b -> p if tot_joint_area_share >= 0.5 &  max_joint_area_share >= 0.7
bp = bp.loc[(bp['tot_joint_area_share'] >= 0.5) & \
            (bp['max_joint_area_share'] >= 0.7) & \
            (bp['joint_area'] == bp['max_joint_area']),\
            ['full_id', 'area_pct', 'n_vertices', 'blklot']]

# find the number of matched buildings for each parcel
bp['blds_per_prcl'] = bp.groupby(['blklot'])['full_id'].transform('nunique')

# if blds_per_prcl > 1 then exclude buildings that are below 1% percentile
bp = bp.loc[(bp['blds_per_prcl'] == 1) | \
            ((bp['blds_per_prcl'] > 1) & \
            (bp['area_pct'] >= 0.01))]
# compute number of buildings matched per parcel after the smallest ones are dropped
bp['blds_per_prcl'] = bp.groupby(['blklot'])['full_id'].transform('nunique')
# keep only 1-to-1 matches
blds_prcl_shrt = bp.loc[bp['blds_per_prcl'] == 1, ['full_id', 'blklot']]


#=======================================================================
# Matching step 3: combine blds_prcl and addr_prcl; check for duplicated addresses
## downloads osm-ways/relations
# ... and add addr:* tags onto them
#=======================================================================
blds_addr = addr_prcl_shrt.merge(blds_prcl_shrt, on = 'blklot', how = 'inner')

#### Check for duplicated addresses
unique_addr = blds_addr[['full_id','full_street_name', 'joint_address_nu', 'joint_address_2']]
# append existing in OSM addresses
osm_addr = blds.loc[(blds['addr:street'].notna() & blds['addr:housenumber'].notna()),\
    ['full_id', 'addr:street', 'addr:housenumber', 'addr:unit']]
#... replace NaN with ""
osm_addr['addr:unit'] = osm_addr.loc[osm_addr['addr:unit'].isna(), 'addr:unit'] = ""

osm_addr.rename(columns={'addr:street':'full_street_name',\
                        'addr:housenumber':'joint_address_nu',\
                        'addr:unit':'joint_address_2'},\
                        inplace=True)
unique_addr = unique_addr.append(osm_addr)

# replace possible "," in added OSM addresses
unique_addr['joint_address_nu'] = unique_addr['joint_address_nu'].str.replace(",", ";")
unique_addr['joint_address_2'] = unique_addr['joint_address_2'].str.replace(",", ";")
# split ";"-separated housenumbers into distinct rows
unique_addr_un = unique_addr['joint_address_2'].str.split(';').apply(pd.Series, 1).stack()
unique_addr_un.index = unique_addr_un.index.droplevel(-1) # to line up with df's index
unique_addr_un.name = 'joint_address_2'

unique_addr_hs = unique_addr['joint_address_nu'].str.split(';').apply(pd.Series, 1).stack()
unique_addr_hs.index = unique_addr_hs.index.droplevel(-1) # to line up with df's index
unique_addr_hs.name = 'joint_address_nu'
unique_addr.drop(['joint_address_nu', 'joint_address_2'], axis = 1, inplace = True)
unique_addr = unique_addr.join(unique_addr_hs)
unique_addr = unique_addr.join(unique_addr_un)

# check if the same "street + housenumber" are assigned to different full_id's
unique_addr.drop_duplicates(subset=['full_id', 'full_street_name', 'joint_address_nu', 'joint_address_2'],\
    keep='first', inplace=True)
dup_join_to_bld = unique_addr.loc[unique_addr.duplicated(subset= ['full_street_name', 'joint_address_nu', 'joint_address_2'],\
    keep=False),:] #identify records that give same addresses to different buildings
# save the duplicated addresses to csv -- for manual check later!
dup_join_to_bld.to_csv(path + "/dup_join_to_bld.csv", index = None, header=True)

# exclude duplicates -- those in dup_join_to_bld -- for manual check/import
unique_addr = unique_addr.loc[~unique_addr['full_id'].isin(dup_join_to_bld['full_id'])]
# remove the existing osm addresses (that were attached for the check earlier)
unique_addr = unique_addr.loc[~unique_addr['full_id'].isin(osm_addr['full_id'])]
# subset the blds_addr based on unique_addr
blds_addr_unique = blds_addr.loc[blds_addr['full_id'].isin(unique_addr['full_id'])]

# Filter buildings, address points and streets out of the entire OSM file for SF
# ... keep only those w/o addr:*
# Input -- planet_-122.537_37.704_878a8cda.osm -- downloaded manually from BBBike
filter_buildings_command = "osmfilter " + path + "planet_-122.538,37.706_-122.337,37.833.osm --keep-nodes= --keep-relations='building=' --keep-ways='building=' --drop='addr:*=' -o=" + path + "buildings_sf_no_addr.osm"
os.system(filter_buildings_command)

# parse the osm-file with buildings as a tree
tree = ET.parse(path + "/buildings_sf_no_addr.osm")
root = tree.getroot()

# do some extra filtering...
# ... remove very long addr:housenumber and addr:unit (process them later separately)
# ... skip/drop "0" from addr:housenumber
matched_mgis = blds_addr_unique
matched_mgis = matched_mgis.loc[(matched_mgis['joint_address_nu'].str.len() <= 255) &\
                                (matched_mgis['joint_address_2'].str.len() <= 255) & \
                                (matched_mgis['joint_address_nu'] != "0")]
# replace (possible) addr:unit that start with ";" with a unit w/o ";" (i.e. replace ";A" with "A")
matched_mgis.loc[matched_mgis['joint_address_2'].str[0] == ';', 'joint_address_2'] = \
    matched_mgis.loc[matched_mgis['joint_address_2'].str[0] == ';', 'joint_address_2'].str[1:]

# Loop over all matched buildings from [matched_mgis]
NN = len(list(matched_mgis['full_id'])) # total number of buildings with newly matched addresses
col_full_id = matched_mgis.columns.get_loc('full_id')
col_street = matched_mgis.columns.get_loc('full_street_name')
col_house = matched_mgis.columns.get_loc('joint_address_nu')
col_unit = matched_mgis.columns.get_loc('joint_address_2')
for v in range(len(matched_mgis['full_id'])):
    building_id = matched_mgis.iloc[v, col_full_id]
    print("Processing building " + str(v) + " out of " + str(NN))
    cur_street, cur_housenum, cur_unit = matched_mgis.iloc[v, \
            [col_street, col_house, col_unit]]
    # find the current building as an element in the parsed OSM XML-tree
    way = []
    if building_id[0]=='w':
        way = root.find("./way[@id='"+building_id[1:]+"']")
    elif building_id[0]=='r':
        way = root.find("./relation[@id='"+building_id[1:]+"']")
    else:
        way = root.find("./node[@id='"+building_id[1:]+"']")
    
    # assign the addr:* tags to the building 
    if str(type(way))=="<class 'xml.etree.ElementTree.Element'>":
        way.set('action', 'modify')
        # addr:street
        street_tag = ET.SubElement(way, "tag")
        street_tag.set('k', 'addr:street')
        street_tag.set('v', cur_street)
        # addr:housenumber
        housenumber_tag = ET.SubElement(way, "tag")
        housenumber_tag.set('k', 'addr:housenumber')
        housenumber_tag.set('v', cur_housenum)
        # add addr:unit if necessary
        if not cur_unit=='':
            unit_tag = ET.SubElement(way, "tag")
            unit_tag.set('k', 'addr:unit')
            unit_tag.set('v', cur_unit)
        tree = ET.ElementTree(root)
        tree = ET.ElementTree(root)

# save the processed tree as OSM-file
tree.write(path + "buildings_with_added_addr.osm", encoding = 'UTF-8')

# filter the resulting file again, to keep only ways/relations with added addr:* tags
filter_buildings_command = "osmfilter " + path + "buildings_with_added_addr.osm --keep-nodes= --keep-relations='addr:*=' --keep-ways='addr:*=' -o=" + path + "open_in_JOSM_and_upload.osm"
os.system(filter_buildings_command)

# add manually upload='never' to <osm > at the beginning of the "open_in_JOSM_and_upload.osm"-file
