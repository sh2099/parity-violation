import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord, ICRS
from astropy.io import fits
import hydra


def import_data(fits_file, random=False):
    """Extract the data from a fits file"""
    hdulist_cmass = fits.open(fits_file)
    # Get data from table
    indices = np.random.choice(int(2e7), 700000, replace=False)
    get_data = hdulist_cmass[1].data[indices]
    ra_cmass = get_data['RA']
    dec_cmass = get_data['DEC']
    z_cmass = get_data['Z']
    w_tot = get_weights(get_data, random)
    print("Weights retrieved")
    hdulist_cmass.close(fits_file)
    # Transform ra, dec into degree coords
    galaxy_coords = SkyCoord(ra=ra_cmass * u.degree, dec=dec_cmass * u.degree, frame=ICRS)

    return galaxy_coords, z_cmass, w_tot


def get_weights(data, random=False):
    wfkp_cmass = np.array(data['WEIGHT_FKP'])
    if not random:
        wsee_cmass = np.array(data['WEIGHT_SEEING'])
        wstar_cmass = np.array(data['WEIGHT_STAR'])
        wnoz_cmass = np.array(data['WEIGHT_NOZ'])
        wcp_cmass = np.array(data['WEIGHT_CP'])
        wsys_cmass = wsee_cmass * wstar_cmass
        w_cmass = wfkp_cmass * wsys_cmass * (wnoz_cmass + wcp_cmass - np.ones(len(wcp_cmass)))
    #weights = np.stack((w_cmass, wfkp_cmass, wstar_cmass, wsee_cmass, wnoz_cmass, wcp_cmass), axis=1)
        w_gal = wfkp_cmass * wstar_cmass * (wnoz_cmass + wcp_cmass - np.ones(len(wcp_cmass)))
        w_group = wsee_cmass
        return w_gal * w_group
    else:
        return wfkp_cmass


def customise_data(ra, dec, z, weight):
    """Filter unwanted z data"""
    # Restrict data range as done by Hou et al
    filtered_z = z[(z>=0.43) & (z<=0.7)]
    filtered_ra = ra[(z>=0.43) & (z<=0.7)]
    filtered_dec = dec[(z>=0.43) & (z<=0.7)]  
    filtered_weight = weight[(z>=0.43) & (z<=0.7)] 
    if len(filtered_z) == len(filtered_dec) == len(filtered_ra) == len(filtered_weight):
        print(len(filtered_z))
        # Transform ra, dec into degree coords
        galaxy_coords = SkyCoord(ra=filtered_ra * u.degree, dec=filtered_dec * u.degree, frame=ICRS)
        return galaxy_coords, filtered_z, filtered_weight
    else:
        print("Broadcasting error")


def normalize_redshift(redshift):
    """Normalize redshift values"""
    return (redshift - np.min(redshift)) / (np.max(redshift) - np.min(redshift))

