import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.path as Path
from astropy import units as u
from astropy.coordinates import SkyCoord, ICRS
from astropy.io import fits
from matplotlib.colors import Normalize
import os
import json
from matplotlib.colors import LinearSegmentedColormap
#import h5py
from scipy.stats import special_ortho_group, ortho_group

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



def random_sampling_overlap(ra, dec, redshift, w_tot, num_samples=1, square_size=1.0, existing_squares=[], type='test'):
    """ Method to generate random samples of the data"""
    norm_z = normalize_redshift(redshift)
    squares = []
    image_arrays = []
    scale_factors = []
    for i in range(num_samples):
        # Randomly select a point ('x') from the dataset
        x_index = np.random.choice(len(ra))
        x_coords = np.array([[ra[x_index], dec[x_index]]])

        # Generate a random orientation 'phi' in [0, 2*pi]
        phi = np.random.uniform(0, 2*np.pi)

        # Select points within the rotated square of dimensions n*n
        vertices = generate_non_overlapping_square(ra, dec, x_coords, square_size, phi, existing_squares)
        squares.append(vertices)
        selected_indices = get_points_in_square(ra, dec, vertices)
        #num_points = np.append(num_points, len(selected_indices[0]))

        # Rotate coordinates by -phi so aligned with axis - creates better plots
        rotated_coords = rotate_coordinates(np.array([ra[selected_indices], dec[selected_indices]]).T,
                                            x_coords, -phi)
        
        img_filename = f'all_images/2025_test_{type}/{type}_1/{type}_{i}'
        bw_filename = f'all_images/pixelated_demo_bw_{type}/{type}_1/{type}_{i}'
        # Save galaxy positions and weights to JSON file
        ######image_weight = compute_image_weighting(w_gal, w_group, selected_indices)
        os.makedirs(os.path.dirname(img_filename), exist_ok=True)
        #os.makedirs(os.path.dirname(bw_filename), exist_ok=True)
        image_array, scale_factor = create_image(rotated_coords[:, 0], rotated_coords[:, 1], norm_z[selected_indices], w_tot[selected_indices], img_filename, square_size)
        #bw_array, bw_scale_factor = create_image(rotated_coords[:, 0], rotated_coords[:, 1], norm_z[selected_indices], w_tot[selected_indices], bw_filename, square_size, bw=True)
        image_arrays.append(image_array)
        scale_factors.append(scale_factor)
        if (i + 1) % 1000 == 0:
            print(f'{i+1}th image created')
    #save_images_to_hdf5(image_arrays, f'all_images/arrays_demo_{type}.h5')
    return squares, scale_factors


def generate_non_overlapping_square(ra, dec, x_coords, square_size, phi, existing_squares):
    """Generate a non-overlapping square"""
    overlap = True
    while overlap:
        # Generate a square
        vertices = generate_square(x_coords, square_size, phi)

        # Check for overlap with existing squares
        overlap = any(check_overlap(vertices, existing) for existing in existing_squares)

        # If overlap is detected, regenerate the square
        if overlap:
            x_index = np.random.choice(len(ra))
            x_coords = np.array([[ra[x_index], dec[x_index]]])
            phi = np.random.uniform(0, 2*np.pi)
        """
        # Use the following line to stop squares in a given set from overlapping
        else:
            existing_squares.append(vertices)
        """

    return vertices


def check_overlap(square1, square2):
    """Check if two squares overlap"""
    x_overlap = ((max(square1[:, 0]) >= min(square2[:, 0])) and (min(square1[:, 0]) <= max(square2[:, 0])))
    y_overlap = ((max(square1[:, 1]) >= min(square2[:, 1])) and (min(square1[:, 1]) <= max(square2[:, 1])))
    return (x_overlap and y_overlap)


def generate_square(x_coords, square_size, phi=0):
    """Generate and rotate the square"""
    # Calculate the coordinates of the corners of the rotated square
    delta = square_size / 2
    relative_vertices = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    vertices = (relative_vertices * delta) + np.array(x_coords)

    # Rotate the square corners by 'phi'
    rotated_corners = np.array(rotate_coordinates(vertices, x_coords, phi))

    return rotated_corners


def rotate_coordinates(coords, centre_coords, phi):
    """ Method to rotate 2d coordinates about their centre point """
    # Tranpose to set centre as origin
    coords_wrt_centre = coords - centre_coords
    # Define 2d rotation matrix
    ### Note for actual implementation prob define outside of method
    rotation_matrix = np.array([
        [np.cos(phi), -np.sin(phi)],
        [np.sin(phi), np.cos(phi)],
    ])

    # Rotate the coordinates
    rotated_coords = [np.dot(rotation_matrix, y) for y in coords_wrt_centre]
    # Transpose coordinates back to original origin
    reboosted_coords = rotated_coords + centre_coords

    return reboosted_coords


def get_points_in_square(ra, dec, square):
    """ Select the points from withi the square's boundaries"""
    path = Path.Path(square)
    points = np.column_stack((ra, dec))
    selected_indices = np.where(path.contains_points(points))
    selected_indices

    return selected_indices

# Function to compute color based on z
def compute_color(z):
    R = np.array([1, 0, 0])
    B = np.array([0, 0, 1])
    return z * R + (1 - z) * B
    

    
def create_image(ra, dec, redshifts, weights, filename, sq_size=1.0, img_size=64, bw=False):
    """Create coloured images by redshift"""
    image = np.zeros((img_size, img_size, 3), dtype=np.float32)
    """
    norm_ra = (ra - np.min(ra)) / sq_size
    #print(norm_ra)
    norm_dec = (dec - np.min(dec)) / sq_size
    norm_ra = np.clip(norm_ra, 0, 1)
    norm_dec = np.clip(norm_dec, 0, 1)
    """
    norm_ra = normalize_redshift(ra)
    #print(norm_ra.max(), norm_ra.min())
    norm_dec = normalize_redshift(dec)
    #print(norm_dec.max(), norm_dec.min())
    if not bw:
        norm_z = normalize_redshift(redshifts)
    else:
        norm_z = np.ones_like(redshifts)
    x_coords = (norm_ra * (img_size-1)).astype(int)
    y_coords = (norm_dec * (img_size-1)).astype(int)
    #print(x_coords.max(), y_coords.max())
    #print(x_coords.min(), y_coords.min())
    # Accumulate weighted colors in the image
    try:
        for x, y, z, weight in zip(x_coords, y_coords, norm_z, weights):
            color = compute_color(z)
            image[x, y] += 10 * weight * color
    except IndexError:
        print("IndexError: x or y coordinates out of bounds")
        #count += 1
        #print(f"Count: {count}")

    normalized_image = np.clip(image / image.max() * 255, 0, 255).astype(np.uint8)
    scale_factor = 255 / image.max()
    # Display the image without surrounding background
    fig, ax = plt.subplots(figsize=(img_size, img_size), dpi=1)
    ax.imshow(normalized_image, interpolation='nearest')
    ax.axis('off')  # Turn off the axis
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
    # To save the image without padding
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

    return image, scale_factor



def diplay_sample_dist(ra, dec, train_data, test_data):
    fig,ax = plt.subplots(figsize=(9, 6))
    #ax.set_xlim([100,270])
    #ax.set_ylim([-10,70])
    # Plot all data
    ax.scatter(ra, dec, color='green', marker='o', s=0.01)
    # Plot train_data in red
    for square in train_data:
        p = Polygon(square, edgecolor='r', fill=False)
        ax.add_patch(p)
    # Plot test_data in black
    for square in test_data:
        p = Polygon(square, edgecolor = 'k', fill=False)
        ax.add_patch(p)
    plt.savefig('2025_test.png')
    plt.show()

"""
def save_images_to_hdf5(image_arrays, filename):
    """"""Save list of image arrays to HDF5 file.""""""
    with h5py.File(filename, 'w') as hf:
        for i, image_array in enumerate(image_arrays):
            hf.create_dataset(f'image_{i}', data=image_array)
"""

def generate_helices(radius, pitch, num_points, length, num_helices):
    # Generate helix points
    
    t = np.linspace(-np.pi * L / p, np.pi * L / p, n)
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = (p / (2 * np.pi)) * t
    """
    t = np.linspace(-np.pi, np.pi, n)
    w = L / p
    x = r * np.cos(w*t) * (t + np.pi)
    y = r * np.sin(w*t) * (t + np.pi)
    z = (p / (2 * np.pi)) * w*t
    """
    helix_points = np.vstack((x, y, z)).T  # Shape: (n, 3)
    # Generate random initial points for each helix
    initial_xy = np.random.rand(num_helices, 2) * 100  # Shape: (num_helices, 3)
    initial_z = np.random.rand(num_helices, 1) * 0.1
    initial_points = np.hstack((initial_xy, initial_z))  # Shape: (num_helices, 3)
    # Generate random rotation matrices for each helix
    rotation_matrices = np.array([special_ortho_group.rvs(3) for _ in range(num_helices)])  # Shape: (num_helices, 3, 3)
    
    #handedness = np.random.randint(0, 2, num_helices)
    handedness = np.zeros(num_helices)
    # Modify helix points based on handedness
    helix_points_modified = np.array([helix_points if h == 0 else np.vstack((x, -y, z)).T for h in handedness])

    # Apply the rotation matrices
    all_helix_points = np.einsum('nij,nkj->nki', rotation_matrices, helix_points_modified)   # Shape: (num_helices, n, 3)
    all_helix_points +=  initial_points[:, np.newaxis, :]
    return all_helix_points

def xyz_distribution(x_join, y_join, z_join):
    coords = [x_join, y_join, z_join]
    fig, ax = plt.subplots(3,2, figsize=(10, 18))
    for i in range(3):
        coord = coords[i]
        normalised_distance = normalise_distances(coord)
        print("Distances Normalised")
        colors = map_distance_to_color(normalised_distance)
        print("Colour map created")
        ax[i,0].scatter(coords[(i+1)%3], coords[(i+2)%3], s=1, c=colors)
        ax[i,1].hist(normalised_distance)
    plt.savefig('helix__cube_distribution.png')

def random_rotation_matrix():
    return special_ortho_group.rvs(3)

# Step 2: Compute the center of each group
def compute_center(points):
    return np.mean(points, axis=0)

# Step 3, 4, and 5: Apply the rotation to each group
def rotate_group(group):
    center = compute_center(group)
    translated_points = group - center
    rotation_matrix = random_rotation_matrix()
    rotated_points = np.dot(translated_points, rotation_matrix.T)
    return rotated_points + center



if __name__ == "__main__":