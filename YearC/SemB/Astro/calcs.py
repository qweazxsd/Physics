import astropy.constants as c
import astropy.units as u
import numpy as np

sigma_sb = c.sigma_sb.to("kg / s3 K4")
h0 = 70 * u.km / (u.Mpc*u.s)
a = 4*sigma_sb/c.c
rho_p = c.c**5/(4*c.G**2*c.hbar.to("kg m2 / s"))
t_plank = (rho_p*c.c**3/(4*c.sigma_sb.to("kg / s3 K4")))**0.25
rho_crit = 0.9*1e-29*u.g/u.cm**3

print((c.hbar.to("kg m2/s")/(8e18*u.GeV.to(u.J)*c.c)))
print(np.sqrt(2*c.G*c.hbar/c.c**3).to(u.m))
print(a*t_plank**3/c.k_B.to("kg m2 / s2 K"))

p = 5e4*u.m*(139*u.MeV).to("kg m 2 / s 2")/(c.c*7.8*u.m)
print((p*c.c).to(u.GeV))

print((14e9*u.yr*(a*1e28*u.K**4/(c.c**2*rho_crit.si))**-0.5).to(u.s))
print(((1e12*u.eV).to("kg m 2 / s 2")/c.c**2))
print(14*u.Gyr*(1e12*u.eV/(rho_crit.si*u.m**3*c.c**2).to(u.eV))**-0.5)
print((1e12*u.eV.to("kg m 2 / s 2")/a)**0.25)
t_TeV = (1e12*u.eV).to(u.J)/(2.7*c.k_B)
print((c.k_B*t_TeV/u.MeV.to(u.J))**-2)
print((26.5e60*u.erg).to(u.eV)/(28.4e6*u.eV))
print(4*5.8e65*c.m_p)
print((1e-20*u.g/u.cm**3).si/c.m_p/(1e9/u.m**3))
print((c.hbar*(7*u.GHz).to("1/s")).to(u.eV))
rho_a = c.M_sun/(4*np.pi/3 * c.R_sun**3)
print(((4*np.pi/81)**(1/3) * c.M_sun**(2/3) * c.G * rho_a**(4/3)).cgs)
print((2*3**5/(np.pi*a))**0.5 * (2*c.k_B.to("kg m2 / s2 K")/(c.m_p*c.G**3))**2)
print(((6**5 * c.k_B.to("kg m2 / s2 K")**4/(np.pi*a*c.m_p**4*c.G**3))**0.5)/c.M_sun)
print((c.M_sun*c.c**2/c.L_sun).to("Gyr"))
eg = ((np.pi*7/137)**2 * 2*14*c.m_p*c.c**2/15).to("MeV")
print((eg/(4*c.k_B*1.5e7*u.K))**(1/3) - 2/3)
print((500*u.keV/(4*c.k_B*1.5e7*u.K))**(1/3) - 2/3)
print(3.3e-4*u.W/u.kg * 0.2 * c.M_sun)
print((c.L_sun*3/(50*(u.W/u.m**3)*4*np.pi))**(1/3)/c.R_sun)
sig = (1e-43 * u.cm**2).to("m2")
tau = 3*sig*c.M_sun/(4*np.pi*c.m_p*c.R_sun**2)
print(tau)
