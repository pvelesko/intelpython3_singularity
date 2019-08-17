
# coding: utf-8

# In[1]:


"""
Traversal Methods:
1. Random traversal of nearest neighbors
    Pros:
        no training time
    Cons:
        Low efficiency: ~25% fuel left

Optimizations:
1. Reduced potential breweries to max round trip
2. Neighbor list precompute (memory requirements O(n^2))
"""


# In[2]:


import pandas as pd
import numpy as np
import haversine
import random
import timeit
from IPython.display import clear_output


# In[3]:


def get_coords(a, data):
    lat = data.loc[a]['latitude']
    lon = data.loc[a]['longitude']
    return (lat, lon)

def dist_coord(x,y):
    h = haversine.haversine(x, y)
    return h

def dist_id(x, y, data):
    return dist_coord(get_coords(x, data), get_coords(y, data))

class BeerTour:
    def get_neighbors(self, idx, data):
        # apply lambda to return df containing distances between brewery idx and all other breweries
        # Sorted nearest to furthest
        a = pd.DataFrame(data.apply(lambda x: dist_coord(data.loc[idx], x), axis=1).sort_values())
        a.columns = ["distance to id %d" % idx]
        return a
    
    def __init__(self):
        self.beers = pd.read_csv("csv_data/beers.csv").set_index('brewery_id')
        self.breweries = pd.read_csv("csv_data/breweries.csv").set_index('id')
        self.categories = pd.read_csv("csv_data/categories.csv")
        geocodes = pd.read_csv("csv_data/geocodes.csv").set_index('id')
        self.styles = pd.read_csv("csv_data/styles.csv").set_index('id')
        coords = geocodes[['latitude', 'longitude']]
        
        # Add home location
        home = pd.Series({'latitude': 51.355468, 'longitude': 11.100790 }, name=0)
        coords = coords.append(home)
        self.coords = coords.sort_index()
        
        # reduce candidates to 1/2 fuel
        neigh = self.get_neighbors(0, coords)
        a = neigh[self.get_neighbors(0, coords) < 1000] # TODO fix as param
        a = a.dropna()
        ids = a.index
        coords = coords.loc[ids]
        
        # make neighbor list
        t = timeit.default_timer()
        max_neighbors = len(coords)
        def n_nearest_neigh(idx, data, n = max_neighbors):
            return list(self.get_neighbors(idx, data).index[1:n+1])

        coord_idx = (list(coords.index))
        coord_idx
        neigh_list = []
        for idxs in coord_idx:
            #clear_output()
            #print("Computing neighbor list for id %d" % idxs)
            neigh_list.append((n_nearest_neigh(idxs, coords)))
            
        ndf = pd.DataFrame(neigh_list)
        ndf.index = coord_idx
        coords['neighbors'] = neigh_list
        coords
        mem = len(coords) ** 2 * 8 / 1000
        print("Memory required for neighbor list %dkB" % mem)

        t = timeit.default_timer() - t
        print("Time to compute neighbor list: %fs" % t)
        self.coords = coords
        self.neigh_list = neigh_list
        
        self.pos = 0 # start at HOME node
        self.rew_big = 10
        self.rew_small = 10
        self.rew_big = 10
        
    # DQL 
    def take_action(self, next_node, fuel):
        # Reward outward movement when fuel > 50%
        # Reward inward movement when fuel > 50%
        dist_to_home_current = dist_id(0, self.pos, self.coords) 
        dist_to_home_next = dist_id(0, next_node, self.coords) 
        inward = dist_to_home_current > dist_to_home_next
        self.state = next_node
        if fuel < 0.5 and inward:
            reward = self.rew_big
        else:
            reward = self.rew_small
        return reward




# In[4]:


env = BeerTour()


# In[5]:


# Satisfy requirement to store data into a DB
#from sqlalchemy import create_engine
#engine = create_engine('sqlite://', echo=False)
#env.coords.to_sql('breweries', con=engine, index=False)


# In[6]:


class random_halo():
    def __init__(self, env, max_dist = 2000):
        self.coords = env.coords
        self.max_dist = max_dist
        self.km = 0
        
    def get_next_id(self, curr, visited):          
        neigh = self.coords.loc[curr]['neighbors'] # neighbor list for current id
        try_count = 0     
        # this is where decision happens
        radius = 3 # start with this sized halo
        while(True):
            try_count+=1
            # Bound to coords
            idx = min(random.randint(0,radius-1), len(env.coords))
            next_id = neigh[idx]
            if (next_id not in set(visited)): break
            # number of tries scaling with radius
            if (try_count > radius**2): radius+=1
                
        # Check if next_id is within available range, if not go home
        dist_next = dist_id(curr, next_id, env.coords)
        dist_next_home = dist_id(next_id, 0, env.coords)
        dist_home = dist_id(curr, 0, env.coords)
        
        if dist_next + dist_next_home < (self.max_dist - self.km):
            self.km += dist_next
        else:
            next_id = -1
            self.km += dist_home
        #print("Current ID: %d Next ID: %d Distance Travelled: %f" %(curr, next_id, self.km))
        return next_id


# In[7]:


def fly(max_dist = 2000):
    agent = random_halo(env)
    visited = []
    curr = 0
    reward = 0
    next_id = 0
    while(True):
        visited.append(curr)
        next_id = agent.get_next_id(curr, visited)
        if next_id == -1: break
        curr = next_id
        reward += env.take_action(next_id, agent.km)
    return visited, agent.km, reward


# In[ ]:


max_visited = 0
best_res = ""
t_run = timeit.default_timer()
fly_data = []
for i in range(500):
    t = timeit.default_timer()
    visited, km, reward = fly()
    fly_data.append([visited, km, reward])
    t = timeit.default_timer() - t
    
    # Parse out total unique beer styles collected
    style_ids = env.beers.loc[visited]['style_id'] # all styles visited
    style_ids = style_ids.dropna() # drop beers for which style is unknown
    style_ids = style_ids[style_ids != -1] # drop -1s 
    collected_styles = env.styles.loc[style_ids]
    collected_styles = collected_styles.drop_duplicates()

    res = "Max: %d Visited %d, Travelled: %f Reward: %f Done in %fs" %(max_visited, len(visited), km, reward, t)
    if (i % 50 == 0):
        #clear_output()
        print(res)
    
    # Maximize Beer styles collected
    if len(collected_styles) > max_visited:
        max_visited = len(collected_styles)
        best_res = (visited, km, collected_styles)
        
t_run = timeit.default_timer() - t_run
print("Time: %fs" % t_run)


# In[9]:


l = "Visited %d breweries. Distance Travelled: %f " % (len(best_res[0]), best_res[1])
print(len(l)*"=")
print(l)
styles = list(best_res[2]['style_name'])
print("Beer Styles Collected: %d" % len(styles))
for style in styles:
    print("    => %s" % style)


# In[10]:


fly_df = pd.DataFrame(fly_data)
with open('fly_data.csv', 'a') as f:
    fly_df.to_csv(f, header=False)

