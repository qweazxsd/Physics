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
quarter_waveplate = Thorlabs.KinesisMotor('55278794', scale='stage')
#
Quarter_waveplate_min_power_circ1 = 62
Quarter_waveplate_min_power_circ2 = 153
Quarter_waveplate_max_power_lin = 107


power_25p7 = 0
power_37 = 10
power_40p5 = 20
power_28p2 = 35
power_22p05 = 40
power_14p8 = 45
power_8p73 = 50
power_6p6 = 52
power_4p77 = 54
power_2p63 = 57




start_loc = 4
end_loc = -0.5
num_loc = 3200



# temps = [10,15,20,25,30,35,40,45,50,53,55,57,60,63,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,195,200,210,220,230,240,250,260,270,280,290,300]


#set_temp_measure_lockin.main(Cryostat,5, method='smart',  interval=0.5, temp_range=0.3, duration=2400, fname='Kerr_T_',lockin_samples = 1)

# sleep(1800)

# ####Setting power to 28.2mW*2
# print(half_waveplate.get_position())
# half_waveplate.move_to(35)
# print("moving")
# half_waveplate.wait_move()
# print("done moving")
# print(half_waveplate.get_position())
#


# start_loc = 8
# end_loc = -0.2
# num_loc = 6400
# #
# ##Setting quarter WP to linear
# print(quarter_waveplate.get_position())
# quarter_waveplate.move_to(Quarter_waveplate_max_power_lin)
# print("moving")
# quarter_waveplate.wait_move()
# print(quarter_waveplate.get_position())
#
#
# ###Measure linear polarization
# pump_probe_5_two_lockins.main(f'Kerr_5K_highres_28p2mW_lin', start_loc, end_loc, num_loc,lockin_samples=1)
#
# ##Setting quarter WP to circ1
# print(quarter_waveplate.get_position())
# quarter_waveplate.move_to(Quarter_waveplate_min_power_circ1)
# print("moving")
# quarter_waveplate.wait_move()
# print(quarter_waveplate.get_position())
#
#
# ###Measure circ1 polarization
# pump_probe_5_two_lockins.main(f'Kerr_5K_highres_28p2mW_right_2', start_loc, end_loc, num_loc,lockin_samples=1)
#
# ##Setting quarter WP to circ2
# print(quarter_waveplate.get_position())
# quarter_waveplate.move_to(Quarter_waveplate_min_power_circ2)
# print("moving")
# quarter_waveplate.wait_move()
# print(quarter_waveplate.get_position())
#
#
# ###Measure circ2 polarization
# pump_probe_5_two_lockins.main(f'Kerr_5K_highres_28p2mW_left_2', start_loc, end_loc, num_loc,lockin_samples=1)
#


temps = [10,15,20,25,30,35,40,45,50,53,55,57,60,63,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,195,200,210,220,230,240,250,260,270,280,290,300]


start_loc = 8
end_loc = -0.2
num_loc = 6400

for temp in temps:

    set_temp.main(Cryostat, temp, method='smart', interval=0.5, temp_range=0.1, duration=180, fname=str(temp))
    sleep(300)

    # pump_probe_5_two_lockins.main(f'Kerr_sample3_spot3_highres_{temp}K_19p7mW', start_loc, end_loc, num_loc,lockin_samples=1)

    ##Setting quarter WP to linear
    print(quarter_waveplate.get_position())
    quarter_waveplate.move_to(Quarter_waveplate_max_power_lin)
    print("moving")
    quarter_waveplate.wait_move()
    print(quarter_waveplate.get_position())

    ###Measure linear polarization
    pump_probe_5_two_lockins.main(f'Kerr_sample3_spot3_{temp}K_28p2mW_lin', start_loc, end_loc, num_loc, lockin_samples=1)

    ##Setting quarter WP to circ1
    print(quarter_waveplate.get_position())
    quarter_waveplate.move_to(Quarter_waveplate_min_power_circ1)
    print("moving")
    quarter_waveplate.wait_move()
    print(quarter_waveplate.get_position())

    ###Measure circ1 polarization
    pump_probe_5_two_lockins.main(f'Kerr_sample3_spot3_{temp}K_28p2mW_right', start_loc, end_loc, num_loc, lockin_samples=1)

    ##Setting quarter WP to circ2
    print(quarter_waveplate.get_position())
    quarter_waveplate.move_to(Quarter_waveplate_min_power_circ2)
    print("moving")
    quarter_waveplate.wait_move()
    print(quarter_waveplate.get_position())

    ###Measure circ2 polarization
    pump_probe_5_two_lockins.main(f'Kerr_sample3_spot3_{temp}K_28p2mW_left', start_loc, end_loc, num_loc, lockin_samples=1)


set_temp.main(Cryostat, 5, method='smart', interval=0.5, temp_range=0.1, duration=180, fname=str('5'))
sleep(300)

sleep(5)
Cryostat.disconnect()
Cryostat.end()
print("Disconnected from cryostat")
