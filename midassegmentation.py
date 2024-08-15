import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import closing
import skimage.morphology
from cellpose import models
from PIL import Image
import scipy.ndimage
from skimage.segmentation import clear_border

# from IPython.display import Markdown
# from google.colab import files
import copy
from matplotlib.colors import ListedColormap

def plottings_structures(combined_layer_full):
    num_labels = len(np.unique(combined_layer_full))
    colors_rgba = plt.cm.tab20(np.linspace(0, 1, num_labels))  # Using Viridis colormap for better color differentiation
    colors_rgba[0, 3] = 0  # Set alpha channel of the background color to 0 (transparent)
    # Create a colormap with transparent background
    cmap_with_transparent_bg = ListedColormap(colors_rgba)
    # Plot the image
    plt.figure()
    plt.imshow(combined_layer_full, cmap='gray')

    # Plot the overlay with different colors for each label and transparent background
    plt.imshow(combined_layer_full, cmap=cmap_with_transparent_bg, alpha=1, interpolation='nearest')
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

# Note that plt.show() is not called inside the function, since it will often be plotted with others on top.

def merge_labels(layers):
    # Convert layers to a NumPy array if it is a list
    if isinstance(layers, list):
        layers = np.array(layers)

    # Ensure layers is a 3D array (number of layers, height, width)
    if len(layers.shape) == 4:
        layers = layers[..., 0]  # Remove the channel dimension if it exists

    # Initialize combined_layer_full with the first layer
    combined_layer_full = layers[0, :, :]

    for layer in layers[1:, :, :]:
        # Find label ids to remove from current layer
        label_id_to_remove = np.unique(layer[combined_layer_full > 0])

        # Get last labels
        last_label = np.max(combined_layer_full)

        # Remove labels from layer and reassign current labels
        for label_id in np.unique(layer)[1:]:
            if label_id in label_id_to_remove:
                layer[layer == label_id] = 0
            else:
                layer[layer == label_id] += last_label

        # Merge current layer into the combined_layer_full
        combined_layer_full += layer

    return combined_layer_full


def ccd_largest_part(img, keep_continuous=False):
    res = np.zeros(img.shape)

    all_unique_labels = np.unique(img)

    for cur_lab, i in zip(range(len(all_unique_labels)), all_unique_labels):
        if i == 0: continue

        labels_out, num_labels = scipy.ndimage.label((img == i).astype(np.uint16),
                                                     structure=np.ones((3, 3))
                                                     )
        if num_labels == 1:
            lab = 1

        else:
            lab_list, lab_count = np.unique(labels_out, return_counts=True)
            if lab_list[0] == 0:
                lab_list = lab_list[1:]
                lab_count = lab_count[1:]

            largest_ind = np.argmax(lab_count)
            lab = lab_list[largest_ind]

        if keep_continuous:
            res += (cur_lab * (labels_out == lab)).astype(np.uint16)
        else:
            res += (i * (labels_out == lab)).astype(np.uint16)

    res = res.astype(np.uint16)

    return res


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img_file_path = "/Users/px/Downloads/SDU x KU - Droplet & foam quantification/Images/Light microscopy/Emulsion droplets (light microscopy)/BF-Droplet (1).tiff"  # @param {type:"string"}
    img_to_seg = np.array(Image.open(img_file_path))

    model_file = "/Users/px/Downloads/SDU x KU - Droplet & foam quantification/Napari-plugin code for droplet quantification/S-Seg/S_seg/food_model_cyto_2_II"  # @param {type:"string"}
    #@markdown ###Item size range to detect: (px):
    smallest = 10 #@param {type:"number"}
    largest = 250 #@param {type:"number"}
    
    #@markdown ###Segmentation steps:
    amount_of_steps = 5 #@param {type:"number"}
    
    #@markdown ###Non-pickyness (flow threshold):
    non_pickyness = 0.4 #@param {type:"number"}
    
    #@markdown ###Color channels to segment:
    channel_greyscale = True #@param {type:"boolean"}
    channel_red = False #@param {type:"boolean"}
    channel_green = False #@param {type:"boolean"}
    channel_blue = False #@param {type:"boolean"}
    channels_to_segment = {'Greyscale': channel_greyscale, 'Red': channel_red, 'Green': channel_green, 'Blue': channel_blue}
    
    #@markdown ###Delete border labels:
    delete_border_labels = False #@param {type:"boolean"}

    try:
        model = models.CellposeModel(pretrained_model=model_file, gpu=True)
       #model = models.CellposeModel(model_type="cyto3", gpu=True) for cyto3 or 2.
    except:
        print("Failed to load pretrained model, please place verify that the provided path is correct")
        print("Using default cyto3 model instead")
        model = models.Cellpose(model_type='cyto3', gpu=True)

    local_img = img_to_seg # this is unnecessary, but just kept here for further check

    layers = []
    count = 1
    total_steps = amount_of_steps
    even_dist_nums = list(np.linspace(smallest, largest, total_steps, dtype=int))
    for i in even_dist_nums:
        print(f"Running cellpose ({count}/{total_steps})", flush=True)
        count += 1
        if channels_to_segment['Greyscale']:
            output = model.eval(local_img, diameter=i, channels=[0,0], flow_threshold=non_pickyness)
            layers.append(output[0])
        if channels_to_segment['Red']:
            output = model.eval(local_img, diameter=i, channels=[1,0], flow_threshold=non_pickyness)
            layers.append(output[0])
        if channels_to_segment['Green']:
            output = model.eval(local_img, diameter=i, channels=[2,0], flow_threshold=non_pickyness)
            layers.append(output[0])
        if channels_to_segment['Blue']:
            output = model.eval(local_img, diameter=i, channels=[3,0], flow_threshold=non_pickyness)
            layers.append(output[0])

    print("Segmentation done\n", flush=True)
    # to numpy array for fast process
    layers = np.array(layers)

    np.save('layers_before_combine.npy', layers)

    # layers = np.load('layers_before_combine.npy')

    plottings_structures(np.sum(layers, axis=0))
    plt.title('combined_raw')
    plt.show()

    combined_layer_full = merge_labels(layers)

    plottings_structures(combined_layer_full)
    plt.title('combined_remove_dups')
    plt.show()

    # Closing operation
    print("Closing operation...")
    combined_layer_full = closing(combined_layer_full)

    # Delete border
    if delete_border_labels:
        print("Deleting border labels...")
        combined_layer_full = clear_border(combined_layer_full)

    plt.imshow(local_img, cmap='gray')
    plt.imshow(combined_layer_full, alpha=0.5)
    plt.show()

    plottings_structures(combined_layer_full)
    plt.imshow(local_img, cmap='gray')
    plt.show()

    plt.savefig('img_ontop.svg', format='svg')
    print("\n\n\n", flush=True)

########################### each label should be a single connected component ###########################
##### small cells are also a bit concerning ###########################

    min_size = 25

    combined_layer_full = skimage.morphology.remove_small_objects(combined_layer_full, min_size=min_size,
                                                                  connectivity=1)
    combined_layer_full = ccd_largest_part(combined_layer_full, keep_continuous=True)

    plottings_structures(combined_layer_full)
    plt.title('combined_keep_largest')
    plt.show()

    np.save('combined_layer_full.npy', combined_layer_full)


    # import PIL.Image as Image
    #
    # im = Image.fromarray(np.moveaxis(np.array(layers), 0, -1))
    # im.save('test.tif')
    #
    # im = Image.fromarray(layers[0])
    # im.save('test0.tif')
    #
    # np.unique(layers[2])
