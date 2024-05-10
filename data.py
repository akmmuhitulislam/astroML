from astroquery.simbad import Simbad
from astroquery.ipac.ned import Ned
from astroquery.exceptions import NoResultsWarning
import warnings
import os

# Add fields to be queried from Simbad
Simbad.add_votable_fields('z_value', 'otype', 'flux(B)')

# Define range of NGC IDs to query
start_id = 1
end_id = 9999

ids = []
images = []
dups = []
labels = []
dup_labels = []

for ngc_id in range(start_id, end_id + 1):
    main_id = f"NGC {ngc_id}"
    try:
        # Query Ned images for the NGC ID
        ned_images = Ned.get_images(main_id)
        
        # If there are images available for the NGC ID
        if len(ned_images) != 0:
            ids.append(main_id)
            
            # Query information from Simbad for the NGC ID
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', NoResultsWarning)
                    sim_des = Simbad.query_object(main_id)
                
                if sim_des is not None:
                    redshift = sim_des['Z_VALUE'].data[0]
                    obj_type = sim_des['OTYPE'].data[0]
                    magnitude_B = sim_des['FLUX_B'].data[0]
                else:
                    redshift = "N/A"
                    obj_type = "N/A"
                    magnitude_B = "N/A"
                
                ned_des = Ned.query_object(main_id)
                if ned_des is not None:
                    velocity = ned_des['Velocity'].data[0]
                else:
                    velocity = "N/A"

                # Check if more than two values are "N/A" in the label
                na_count = sum(value == "N/A" for value in [redshift, velocity, obj_type, magnitude_B])
                if na_count <= 2:
                    label = f"Redshift (z_value) = {redshift}\n"
                    label += f"Velocity = {velocity}\n"
                    label += f"Object Type = {obj_type}\n"
                    label += f"Flux Magnitude (B) = {magnitude_B}\n"

                    images.append(ned_images[0])
                    labels.append(label)
                    if len(ned_images) > 1:
                        for img in ned_images[1:]:
                            dups.append(img)
                            dup_labels.append(label)
                else:
                    print(f"Skipping NGC {ngc_id}: More than two values are 'N/A'")
                    # Remove the NGC ID and images corresponding to this ID
                    ids.pop()
                
            except Exception as e:
                print(f"Error processing NGC {ngc_id}: {e}")
                # Append empty label if query fails
                labels.append("")
        else:
            print(f"No images found for {main_id}")
    except:
        print(f"NGC {main_id} 404 NOT FOUND")

print(f'Total IDs: {len(ids)}\nTotal Images: {len(images)}')

# Save images and labels
for i, (img, label, ngc_id) in enumerate(zip(images, labels, ids)):
    # Save image
    try:
        img.writeto(f'images/{ngc_id}_image.fits')
    
        # Save label
        with open(f'labels/{ngc_id}_label.txt', 'w') as f:
            f.write(label)
    except:
        print('could save an image.')

# Save duplicate images and labels
for i, (img, label, ngc_id) in enumerate(zip(dups, dup_labels, ids)):
    # Save image
    try:
        img.writeto(f'duplicates/images/{ngc_id}_image_{i+1}.fits')
    
        # Save label
        with open(f'duplicates/labels/{ngc_id}_label_{i+1}.txt', 'w') as f:
            f.write(label)
    except:
        print('could save an image.')