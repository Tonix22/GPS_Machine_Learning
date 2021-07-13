import math

winds = ["E", "SEE", "SE", "SSE", "S", "SSW", "SW", "SWW", "W", "NWW", "NW", "NNW","N", "NNE", "NE", "NEE"]
#360/16*n
def winds_to_degree(sub_s):
    for index, string in enumerate(winds):
         if sub_s in string:
            return index*22.5 #360/16

def plot_speed_and_angle(radius,angle):
    x = radius*math.cos(angle)
    y = radius*math.sin(angle)
    return x,y

winds_to_degree("SW")