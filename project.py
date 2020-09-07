from sgp4.api import Satrec
from sgp4.api import jday

lk_coord = '55.930148, 37.518151'

s = '1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991'
t = '2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482'
satellite = Satrec.twoline2rv(s, t)

jd, fr = jday(2020, 9, 9, 12, 00, 0)

# semimajor = satellite.a * satellite.radiusearthkm
# eccentricity = satellite.ecco
# inclination = satellite.inclo
# argument_periapsis = satellite.argpo
# ascending_node = satellite.Om
# mean_anomaly = satellite.mo
# gravitational_parameter = satellite.mu

result = satellite.sgp4(jd, fr)

print(result)

# satellite_params = [semimajor, eccentricity, inclination, argument_periapsis, ascending_node, mean_anomaly,
#                     gravitational_parameter]

# print(satellite_params)

# true_anomaly = mean_anomaly_to_true_anomaly_eccentric(mean_anomaly, eccentricity)
#
# print(oetoeci(gravitational_parameter, semimajor, eccentricity, inclination, argument_periapsis, ascending_node,
#               true_anomaly))







