import numpy as np
Cam1 = {0,0}
Cam2= [0.2, -0.2]
Pi_2 = np.pi/2
Xmin = 0.2
Ymin = 0.2
tetha = 0 

R1 = np.sqrt(Xmin**2 + Ymin**2)
R2 = np.sqrt((Xmin+Cam2[0])**2 + (Ymin+Cam2[1])**2)
# R1*np.sin(tetha) =R2 np.cos(tetha) + 0.2
# Ymin*np.cos(tetha) = Ymin*np.sin(tetha) - 0.2

# Xmin*(np.cos(tetha-Pi_2) - np.cos(tetha)) = 0.2

# 0.2/Xmin = np.cos(tetha-Pi_2) - np.cos(tetha)

# 0.2/Xmin = 2*np.sin(-Pi_2)*np.sin(2*tetha-Pi_2)

# 0.2/Xmin = -2*np.sin(2*tetha-Pi_2)

# np.arcsin((0.2/Xmin)/2) = 2*tetha-Pi_2

tetha = ((np.arcsin((Cam2[0]/R1)/2))+Pi_2)/2
print(tetha*360/6.28)