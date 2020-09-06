from typing import List
import numpy as np

from sgp4.api import Satrec

from types import ArrayFloat, ArrayVector3
from utils import mean_anomaly_to_true_anomaly_eccentric

s = '1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991'
t = '2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482'
satellite = Satrec.twoline2rv(s, t)


semilatus = satellite.a
eccentricity = satellite.ecco
inclination = satellite.inclo
argument_periapsis = satellite.argpo
ascending_node = satellite.Om
mean_anomaly = satellite.mo
true_anomaly = mean_anomaly_to_true_anomaly_eccentric(mean_anomaly, eccentricity)
gravitational_parameter = satellite.mu






