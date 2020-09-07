from datetime import datetime

from sgp4.api import Satrec
from sgp4.api import jday

from coordinate_system.EarhRotating import build_default_rotating_cs
from XtoY.cartesianfromgeo import get_cartesian_geo_from_geo_angles
from coordinate_system.earth_models import get_wgs_84

# широта и долгота ЛК
lk_coord = [55.930148, 37.518151]
print("Широта и долгота ЛК", lk_coord)

# декартовы координты ЛК
lk_cartesian = get_cartesian_geo_from_geo_angles(latitude=lk_coord[0], longitude=lk_coord[1], model=get_wgs_84())

print("Декартовы координаты ЛК", lk_cartesian)

s = '1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991'
t = '2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482'
satellite = Satrec.twoline2rv(s, t)

jd, fr = jday(2020, 9, 9, 12, 00, 0)

# получаем векторы координат и скорости спутника в конкретный момент времени
result = satellite.sgp4(jd, fr)

print("Координаты спутника x, y, z", result[1])
print("Скорсть спутника v_x, v_y, v_z", result[2])

# получаем матрицу перехода от ИСО в НеИСО в момент времени data
data = datetime(2020, 9, 7).timestamp()
rotator = build_default_rotating_cs()
rm = rotator.get_matrix(data)

# координаты в нужной системе отсчета
result_in_itrs = rm.dot(result[1])
print("Координаты спутника в НеИСО", result_in_itrs)

lk_in_itrs = rm.dot(lk_cartesian)
print("Координаты ЛК в НеИСО в километрах", lk_in_itrs/1000)