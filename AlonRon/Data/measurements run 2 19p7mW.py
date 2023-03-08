import pump_probe_4
import pump_probe_5_two_lockins
import set_temp_measure_lockin
import set_temp
import numpy as np
from time import sleep, time, strftime
import Attodry.PyattoDRY as Attodry800

from pylablib.devices import Thorlabs

import set_temp_measure_lockin


start_temp = 0
end_temp = 0
num_temp = 1


start_loc = 4
end_loc = -0.5
num_loc = 3200


###Connecting to cryostat
print("Disconnected from cryostat")
Cryostat = Attodry800.Cryostat('COM5')
#
half_waveplate = Thorlabs.KinesisMotor('55276944', scale='stage')
# quarter_waveplate = Thorlabs.KinesisMotor('55278794', scale='stage')
#
# quarter_waveplate_left_max = 58  # Degrees
# quarter_waveplate_right_max = 145  # Degrees
# quarter_waveplate_min = 101  # Degrees

power_13p8 = 0
power_19p7 = 10
power_21 = 20
power_11p26 = 40
power_7p5 = 45
power_4p4 = 50
power_3p8 = 51
power_1p84 = 55




start_loc = 4
end_loc = -0.5
num_loc = 3200
#
# temp=5
#
# set_temp.main(Cryostat, 5, method='smart', interval=0.5, temp_range=0.1, duration=180, fname=str(temp))
# sleep(300)
#
# pump_probe_5_two_lockins.main(f'Kerr_sample3_spot3_{temp}K_7p5mW', start_loc, end_loc, num_loc,lockin_samples=1)
#
# start_loc = 150
# end_loc = -20
# num_loc = 3200
#
# pump_probe_5_two_lockins.main(f'Kerr_sample3_spot3_{temp}K_7p5mW_long', start_loc, end_loc, num_loc,lockin_samples=1)



temps = [10,15,20,25,30,35,40,45,50,53,55,57,60,63,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,195,200,210,220,230,240,250,260,270,280,290,300]


set_temp_measure_lockin.main(Cryostat,5, method='smart',  interval=0.5, temp_range=0.1, duration=180, fname='Kerr_T_',lockin_samples = 1)

sleep(1800)

start_loc = 4
end_loc = -0.5
num_loc = 3200

temp = 5

#######5K Fluences
half_waveplate.move_to(power_19p7)
print("moving")
half_waveplate.wait_move()
print("done moving")
pump_probe_5_two_lockins.main(f'Kerr_sample3_spot3_run2_Fluences_{temp}K_19p7mW', start_loc, end_loc, num_loc, lockin_samples=1)


half_waveplate.move_to(power_13p8)
print("moving")
half_waveplate.wait_move()
print("done moving")
pump_probe_5_two_lockins.main(f'Kerr_sample3_spot3_run2_Fluences_{temp}K_13p8mW', start_loc, end_loc, num_loc, lockin_samples=1)

half_waveplate.move_to(power_11p26)
print("moving")
half_waveplate.wait_move()
print("done moving")
pump_probe_5_two_lockins.main(f'Kerr_sample3_spot3_run2_Fluences_{temp}K_11p26mW', start_loc, end_loc, num_loc, lockin_samples=1)

half_waveplate.move_to(power_7p5)
print("moving")
half_waveplate.wait_move()
print("done moving")
pump_probe_5_two_lockins.main(f'Kerr_sample3_spot3_run2_Fluences_{temp}K_7p5mW', start_loc, end_loc, num_loc, lockin_samples=1)

half_waveplate.move_to(power_4p4)
print("moving")
half_waveplate.wait_move()
print("done moving")
pump_probe_5_two_lockins.main(f'Kerr_sample3_spot3_run2_Fluences_{temp}K_4p4mW', start_loc, end_loc, num_loc, lockin_samples=1)


half_waveplate.move_to(power_3p8)
print("moving")
half_waveplate.wait_move()
print("done moving")
pump_probe_5_two_lockins.main(f'Kerr_sample3_spot3_run2_Fluences_{temp}K_3p8mW', start_loc, end_loc, num_loc, lockin_samples=1)

temps = [5,10,15,20,25,30,35,40,45,50,53,55,57,60,63,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,195,200,210,220,230,240,250,260,270,280,290,300]

half_waveplate.move_to(power_19p7)
print("moving")
half_waveplate.wait_move()
print("done moving")


for temp in temps:

    set_temp.main(Cryostat, temp, method='smart', interval=0.5, temp_range=0.1, duration=180, fname=str(temp))
    sleep(300)

    start_loc = 4
    end_loc = -0.5
    num_loc = 3200

    pump_probe_5_two_lockins.main(f'Kerr_sample3_spot3_run2_{temp}K_19p7mW', start_loc, end_loc, num_loc,lockin_samples=1)

#
#
# set_temp.main(Cryostat, 5, method='smart', interval=0.5, temp_range=0.1, duration=180, fname=str('5'))
# sleep(300)
set_temp_measure_lockin.main(Cryostat, 5, method='smart', interval=0.5, temp_range=0.1, duration=180, fname='Cooldown2_300K_5K_with_polarizers',lockin_samples = 10)

sleep(5)
Cryostat.disconnect()
Cryostat.end()
print("Disconnected from cryostat")
