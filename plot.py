from astropy.io import fits
import matplotlib.pyplot as plt
def load_images_and_labels(num_images):
    images = []
    labels = []
    
    for i in range(3160, num_images + 3160):
        # Load image
        with fits.open(f'test/NGC {i}_image.fits') as hdul:
            image_data = hdul[0].data
            images.append(image_data)
        
        # Load label
        with open(f'testlabels/NGC {i}_label.txt', 'r') as f:
            for line in f:
                if "Object Type" in line:
                    labels.append(line.strip())
                    break  
    
    return images, labels

# Specify the number of images to load
num_images = 2  

# Load images and labels
loaded_images, loaded_labels = load_images_and_labels(num_images)

def display_fits_images(hdu_list, names):
    """
    Display FITS images from a list of HDU objects.
    Args:
        hdu_list: List of HDU objects.
        names: Names of the celestial object to display on the images.
    """
    fig, ax = plt.subplots(1, len(hdu_list), figsize=(5 * len(hdu_list), 8))
    if len(hdu_list) == 1:
        ax = [ax]  # Make single plot handle iterable

    for i, hdu in enumerate(hdu_list):
        # Access the image data from the HDU object
        image_data = hdu.data
        image_shape = image_data.shape
        
        # Display the image data
        ax[i].imshow(image_data, cmap='gray', origin='lower')
        # Update title to include the name and image number
        ax[i].set_title(f"{names[i]} - Image {i} (Shape: {image_shape})")
        ax[i].axis('off')  # Turn off axis numbers and ticks

    plt.tight_layout()
    plt.savefig(f'galaxy_image.png')
    plt.show()
display_fits_images(loaded_images,loaded_labels)