Om     = np.diag(mode_freq)  # GHz Dim(mode, mode)
Ej     = np.diag(Ejs)  # GHz Dim(junctions, junctions)
Ec     = np.diag(Ecs)  # GHz Dim(junctions, junctions)
P      = []  # Dim (mode, junctions)
PC     = []  # Dim (mode, junctions)
Ljs    = []  # Dim (junctions)
Cjs    = []  # Dim (junctions)
I_peak = []  # Dim (mode, junctions)
V_peak = []  # Dim (mode, junctions)
S      = []  # Dim (mode, junctions)
for i, mode in enumerate(modes):
    for j, junc in enumerate(junctions):
        I_peak[i, j], V_peak[i, j] = calc_current_using_line_voltage(
                                                                line[j],  # assosiated with each junction
                                                                Lj[j],
                                                                Cj[j]
                                                                )
        S[i, j] = 1 if V_peak[i, j] > 0 else -1

    U_J_inds = 0.5 * Ljs * I_peak[i, :]**2
    U_J_caps = 0.5 * Cjs * V_peak[i, :]**2

    U_tot_ind = calc_energy_magnetic() / 2 + sum(U_J_inds)
    U_tot_cap = calc_energy_electric() / 2 + sum(U_J_caps)

    U_norm = U_tot_cap  # or (U_tot_ind + U_tot_cap)/2

    P[i, :] = U_J_inds / U_norm
    PC[i, :] = U_J_caps / U_norm


PHI_zpf = S * sqrt(0.5 * Om @ P @ np.linalg.inv(Ej))
CHI = 0.25 * Om @ P @ np.linalg.inv(Ej) @ P.T @ Om * 1000  # MHz
alpha = 0.5 * np.diag(CHI)
Lamb_shift = 0.5 np.trace(CHI)
