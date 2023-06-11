from cosmoprimo.fiducial import DESI


def get_f_reconstruction(zbox, zlim=None, mock_type='cutsky'):
    cosmo = DESI()
    H_0 = 100.0
    f_cubic = cosmo.sigma8_z(z=zbox, of='theta_cb') / cosmo.sigma8_z(z=zbox, of='delta_cb')
    if mock_type == 'cubicbox': return f_cubic
    H_cubic = H_0 * cosmo.efunc(zbox)
    a_cubic = 1 / (1 + zbox)
    zmin, zmax = zlim
    zmid = (zmax + zmin) / 2
    H_cutsky = H_0 * cosmo.efunc(zmid)
    a_cutsky = 1 / (1 + zmid)
    return (a_cubic * H_cubic) / (a_cutsky * H_cutsky) * f_cubic