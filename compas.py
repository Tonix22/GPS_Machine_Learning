import math

#winds = ["E", "SEE", "SE", "SSE", "S", "SSW", "SW", "SWW", "W", "NWW", "NW", "NNW","N", "NNE", "NE", "NEE"]

winds = ["E", 
         "NEE", "NE", "NNE",
         "N", 
         "NNW","NW","NWW",
         "W",
         "SWW","SW","SSW",
         "S",
         "SSE","SE","SEE"]

#360/16*n
def winds_to_degree(sub_s):
    for i in range(0,len(winds)):
        if(sub_s == winds[i]):
            return i*2*math.pi/16 #360/16 but plot lib is in radians     

def plot_speed_and_angle(radius,angle):
    x = radius*math.cos(angle)
    y = radius*math.sin(angle)
    return x,y

winds_to_degree("SW")