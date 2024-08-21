import scipy.ndimage
from scipy.spatial import KDTree
from skimage.measure import find_contours
import skimage
import csv
from matplotlib.colors import ListedColormap
import SimpleITK as sitk
from PIL import Image

from scipy.spatial import distance
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, label
import pandas as pd  # Import pandas to handle CSV files

# from google.colab import files  # Import files from google.colab to handle downloads

def label_statistics(intensity_image: np.ndarray, label_image: np.ndarray, size: bool, intensity: bool, perimeter: bool, shape: bool, position: bool, moments: bool, lstat_scale:bool) -> "pandas.DataFrame":
    def _append_to_column(dictionary, column_name, value):
        if column_name not in dictionary.keys():
            dictionary[column_name] = []
        dictionary[column_name].append(value)

    sitk_label_image = sitk.GetImageFromArray(np.asarray(label_image).astype(int))
    if intensity:
        sitk_intensity_image = sitk.GetImageFromArray(intensity_image)
        intensity_stats = sitk.LabelStatisticsImageFilter()
        intensity_stats.Execute(sitk_intensity_image, sitk_label_image)

    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.SetComputeFeretDiameter(shape)
    shape_stats.SetComputeOrientedBoundingBox(False)
    shape_stats.SetComputePerimeter(perimeter)
    shape_stats.Execute(sitk_label_image)

    results = {}

    for l in shape_stats.GetLabels():
        ##range(1, stats.GetNumberOfLabels() + 1):
        _append_to_column(results, "label", l)

        if intensity:
            _append_to_column(results, "maximum", intensity_stats.GetMaximum(l))
            _append_to_column(results, "mean", intensity_stats.GetMean(l))
            _append_to_column(results, "median", intensity_stats.GetMedian(l))
            _append_to_column(results, "minimum", intensity_stats.GetMinimum(l))
            _append_to_column(results, "sigma", intensity_stats.GetSigma(l))
            _append_to_column(results, "sum", intensity_stats.GetSum(l))
            _append_to_column(results, "variance", intensity_stats.GetVariance(l))

        if position:
            for i, value in enumerate(shape_stats.GetBoundingBox(l)):
                _append_to_column(results, "bbox_" + str(i), value)

            for i, value in enumerate(shape_stats.GetCentroid(l)):
                _append_to_column(results, "centroid_" + str(i), value)

        if shape:
            _append_to_column(results, "elongation", shape_stats.GetElongation(l))
            _append_to_column(results, "feret_diameter [\u03BCm]", shape_stats.GetFeretDiameter(l)*lstat_scale)
            _append_to_column(results, "flatness", shape_stats.GetFlatness(l))
            _append_to_column(results, "roundness", shape_stats.GetRoundness(l))

        if size:
            for i, value in enumerate(shape_stats.GetEquivalentEllipsoidDiameter(l)):
                _append_to_column(results, "equivalent_ellipsoid_diameter_" + str(i) + " [\u03BCm]", value/lstat_scale)
            _append_to_column(results, "equivalent_spherical_perimeter [\u03BCm]", shape_stats.GetEquivalentSphericalPerimeter(l)/lstat_scale)
            _append_to_column(results, "equivalent_spherical_radius [\u03BCm]", shape_stats.GetEquivalentSphericalRadius(l)/lstat_scale)
            _append_to_column(results, "number_of_pixels", shape_stats.GetNumberOfPixels(l)/lstat_scale**2)
            _append_to_column(results, "number_of_pixels_on_border", shape_stats.GetNumberOfPixelsOnBorder(l))

        if perimeter:
            _append_to_column(results, "perimeter [\u03BCm]", shape_stats.GetPerimeter(l)/lstat_scale)
            _append_to_column(results, "perimeter_on_border [\u03BCm]", shape_stats.GetPerimeterOnBorder(l)/lstat_scale)
            _append_to_column(results, "perimeter_on_border_ratio", shape_stats.GetPerimeterOnBorderRatio(l))

        if moments:
            for i, value in enumerate(shape_stats.GetPrincipalAxes(l)):
                _append_to_column(results, "principal_axes" + str(i), value)

            for i, value in enumerate(shape_stats.GetPrincipalMoments(l)):
                _append_to_column(results, "principal_moments" + str(i), value)

    return pd.DataFrame(results)

def rgb2grey(rgb_image):
    return np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140])

def calculate_contours(structure):
    contours = find_contours(structure, level=0.5)
    return contours[0] if contours else None

def show_contours(contours, structures):
    fig, ax = plt.subplots()
    ax.imshow(structures, cmap=plt.cm.gray)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1)
    plt.show()

def calculate_nearest_neighbor(structures):
    labels = np.unique(structures)[1:]  # Exclude background label
    contours = {label: calculate_contours(structures == label) for label in labels}

    nearest_neighbor_distances = {}
    trees = {label: KDTree(contour) for label, contour in contours.items() if contour is not None}

    for label1 in labels:
        min_distance = np.inf
        if label1 not in trees:
            continue
        tree1 = trees[label1]
        for label2 in labels:
            if label1 == label2 or label2 not in trees:
                continue
            tree2 = trees[label2]
            distances, indices = tree2.query(tree1.data, k=1)
            min_dist_idx = distances.argmin()
            min_dist = distances[min_dist_idx]
            if min_dist < min_distance:
                min_distance = min_dist
                min_label = label2
                min_point1 = tree1.data[min_dist_idx]
                min_point2 = tree2.data[indices[min_dist_idx]]
        nearest_neighbor_distances[label1] = (min_label, min_distance, min_point1, min_point2)
    return nearest_neighbor_distances

def plot_structures_with_lines(structures, nearest_neighbor_distances):

    plottings_structures(structures)

    for label, (nearest_label, distance, point1, point2) in nearest_neighbor_distances.items():
        if point1 is not None and point2 is not None:
            plt.plot([point1[1], point2[1]], [point1[0], point2[0]], 'r--')
    plt.title("Segmented Structures with Nearest Neighbor Distances")
    plt.axis('off')
    plt.show()

def plot_histogram_nearest_neighbor_contour(nearest_neighbor_distances):
    distances = [distance for _, (_, distance, _, _) in nearest_neighbor_distances.items()]
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=20, color='darkblue', edgecolor='black', alpha=1)
    plt.xlabel('Nearest Neighbor Distance from Contour [px]')
    plt.ylabel('Count, log10 scale')
    plt.title('Histogram of Nearest Neighbor Distances from centroid')
    plt.grid(True, linestyle='--', linewidth=0.5, zorder=0)
    plt.yscale('log')
    plt.show()

def plot_histogram_nearest_neighbor_centroid(nearest_neighbor_distances):
    distances = [distance for _, (_, distance, _, _) in nearest_neighbor_distances.items()]
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=20, color='darkgreen', edgecolor='black', alpha=1)
    plt.xlabel('Nearest Neighbor Distance from Centroid [px]')
    plt.ylabel('Count, log10 scale')
    plt.title('Histogram of Nearest Neighbor Distances from centroid')
    plt.grid(True, linestyle='--', linewidth=0.5, zorder=0)
    plt.yscale('log')
    plt.show()


def export_to_csv(nearest_neighbor_distances, filename='nearest_neighbor_distances.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Label', 'Nearest Neighbor Label', 'Distance', 'Point1_x', 'Point1_y', 'Point2_x', 'Point2_y'])
        for label, (nearest_label, distance, point1, point2) in nearest_neighbor_distances.items():
            writer.writerow([label, nearest_label, distance, point1[0], point1[1], point2[0], point2[1]])


def func_randomize_labels(combined_layer_full):
    first_value = combined_layer_full[0, 0]  # Get the first value
    unique_labels = np.unique(combined_layer_full)
    unique_labels = unique_labels[unique_labels != first_value]  # Exclude background value
    for label in unique_labels:
        indices_to_randomize = np.where(combined_layer_full == label)
        new_value = np.random.randint(10, 256)
        combined_layer_full[indices_to_randomize] = new_value
    return combined_layer_full

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



def calculate_centroid(structure):
    """
    Calculate the centroid of a segmented structure.

    Args:
    - structure (numpy.ndarray): Array representing the segmented structure.

    Returns:
    - centroid (tuple): Coordinates of the centroid.
    """
    indices = np.transpose(np.nonzero(structure))
    centroid = np.mean(indices, axis=0)
    return tuple(centroid)

def calculate_nearest_neighbor_centroid(structures):
    """
    Calculate the nearest neighbor distance for each unique label in the array.

    Args:
    - structures (numpy.ndarray): Array with multiple structures represented as unique labels.

    Returns:
    - nearest_neighbor_distances (dict): Dictionary containing nearest neighbor distance for each label.
    """
    labels = np.unique(structures)[1:]  # Exclude background label
    centroids = {label: calculate_centroid(structures == label) for label in labels}

    nearest_neighbor_distances = {}
    for label1 in labels:
        min_distance = np.inf
        centroid1 = centroids[label1]
        min_label = label1
        min_centroid2 = centroid1
        for label2 in labels:
            if label1 != label2:
                centroid2 = centroids[label2]
                dist = distance.euclidean(centroid1, centroid2)
                if dist < min_distance:
                    min_distance = dist
                    min_label = label2
                    min_centroid2 = centroid2
        nearest_neighbor_distances[label1] = (min_label, min_distance, centroid1, min_centroid2)

    return nearest_neighbor_distances


def calculate_nearest_neighbor_centroid_w_table(structures, table):
    """
    Calculate the nearest neighbor distance for each unique label in the array.

    Args:
    - structures (numpy.ndarray): Array with multiple structures represented as unique labels.
    - table (pd.DataFrame): previously generated table

    Returns:
    - nearest_neighbor_distances (dict): Dictionary containing nearest neighbor distance for each label.
    """
    labels = np.unique(structures)[1:]  # Exclude background label

    # NOTE that GetCentroid function from simpleitk returns in y, x order.
    centroids = {label: (table.loc[table['label'] == label]['centroid_1'].values[0],
                         table.loc[table['label'] == label]['centroid_0'].values[0]) for label in labels}

    nearest_neighbor_distances = {}
    for label1 in labels:
        min_distance = np.inf
        centroid1 = centroids[label1]
        min_label = label1
        min_centroid2 = centroid1
        for label2 in labels:
            if label1 != label2:
                centroid2 = centroids[label2]
                dist = distance.euclidean(centroid1, centroid2)
                if dist < min_distance:
                    min_distance = dist
                    min_label = label2
                    min_centroid2 = centroid2
        nearest_neighbor_distances[label1] = (min_label, min_distance, centroid1, min_centroid2)

    return nearest_neighbor_distances

def plot_structures_with_lines_centroid(structures, nearest_neighbor_distances):
    """
    Plot segmented structures with lines indicating nearest neighbor distances.

    Args:
    - structures (numpy.ndarray): Array with multiple structures represented as unique labels.
    - nearest_neighbor_distances (dict): Dictionary containing nearest neighbor distance for each label.
    """
    plt.figure(figsize=(8, 6))

    plottings_structures(structures)


    for label, (nearest_label, distance, centroid1, centroid2) in nearest_neighbor_distances.items():
        plt.plot([centroid1[1], centroid2[1]], [centroid1[0], centroid2[0]], 'r--')

    """
    The code below from original notebook is commented out. Showing text just becomes too messy with many labels
    """

    # for label in np.unique(structures):
    #     if label == 0:  # Skip background label
    #         continue
    #     indices = np.where(structures == label)
    #     centroid = (int(np.mean(indices[0])), int(np.mean(indices[1])))
    #     plt.text(centroid[1], centroid[0], str(label), color='red', fontsize=12, ha='center', va='center')

    mask = np.zeros_like(structures) #MASKS FOR ON TOP

    plt.imshow(mask, alpha=0.2)

    plt.title("Segmented Structures with Nearest Neighbor Distances")
    plt.axis('off')
    plt.show()


def contour_analysis_unfold(segmented_labels):
    results = []

    for label in np.unique(segmented_labels):
        if label == 0:
            continue  # Skip background

        # Extract coordinates of the labeled region
        ys, xs = np.where(segmented_labels == label)

        # Calculate center of mass
        cx = np.mean(xs)
        cy = np.mean(ys)

        # Calculate distance and angle for all edge points
        edge_points = []
        for x, y in zip(xs, ys):
            distance = np.sqrt((x - cx)**2 + (y - cy)**2)
            angle = np.arctan2(y - cy, x - cx)
            edge_points.append((distance, angle))

        # Sort edge points by angle
        edge_points.sort(key=lambda x: x[1])

        # Interpolate and sample with fixed theta step
        theta_step = np.pi / 40  # 5 degrees
        sampled_points = []
        prev_theta = None
        for distance, angle in edge_points:
            if prev_theta is None or angle - prev_theta >= theta_step:
                sampled_points.append((distance, angle))
                prev_theta = angle

        # Compute Delta_r between sampled points
        delta_r = []
        for i in range(len(sampled_points) - 1):
            delta_r.append(abs(sampled_points[i + 1][0] - sampled_points[i][0]))

        # Normalize delta_r to unit radius
        max_radius = max(point[0] for point in sampled_points)
        delta_r_normalized = [dr / max_radius for dr in delta_r]

        results.append({
            'label': label,
            'sampled_points': sampled_points,
            'delta_r_normalized': delta_r_normalized
        })

    return results

def curvature(contour):
    dx_dt = np.gradient(contour[:, 0, 0])
    dy_dt = np.gradient(contour[:, 0, 1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / ((dx_dt ** 2 + dy_dt ** 2) ** (3 / 2))
    return curvature
def contour_analysis(combined_layer_full, sigma=0.5, subsample_interval=3, plot=False):
    """
    Analyze the contours of segmented structures and calculate the curvature along them.

    Parameters:
        combined_layer_full: np.array
            The full image containing segmented structures.
        sigma: float, optional
            The standard deviation for the Gaussian filter used to smooth the curvature. Default is 0.5.
        subsample_interval: int, optional
            The interval for subsampling the contour points. Default is 3.
        plot: bool, optional
            If True, plots the segmented structures and their curvature. Default is False.

    Returns:
        pd.DataFrame
            A DataFrame containing the curvature data for all segmented structures.
    """

    # Initialize an empty DataFrame to store all curvature data
    all_curvature_data = pd.DataFrame()

    # Plot curvature for each segmented structure
    for index, label_id in enumerate(np.unique(combined_layer_full)[1:]):

        print(f"Analyzing Segmented Structure {index + 1}")
        segmented_structure = np.where(combined_layer_full == label_id, 1, 0)
        
        # Contour definition from cv2
        contours, _ = cv2.findContours(segmented_structure.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Skip if no contour is found
        if len(contours) == 0:
            print(f"No contour found for Segmented Structure {index + 1}")
            continue

        # Subsample the contour points at regular intervals
        contour_length = len(contours[0])
        indices = np.linspace(0, contour_length - 1, int(np.ceil(contour_length / subsample_interval)))
        contour_subsampled = contours[0][indices.astype(int)]

        if len(contour_subsampled) < 3:
            print('No curvature calculation possible due to insufficient contour points')
            continue

        # Calculate curvature for subsampled contour
        curvatures = curvature(contour_subsampled)
        positions = np.linspace(0, 1, len(curvatures))

        # Smooth the curvature signal using a Gaussian filter
        curvatures_smoothed = gaussian_filter1d(curvatures, sigma=sigma)
        positions_smoothed = np.linspace(0, 1, len(curvatures_smoothed))

        # Append the data to the DataFrame
        data = {'Structure': index + 1, 'Position': positions_smoothed, 'Curvature': curvatures_smoothed}
        df = pd.DataFrame(data)
        all_curvature_data = pd.concat([all_curvature_data, df], ignore_index=True)

        if plot:
            # Plot the results for this segmented structure
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(segmented_structure, cmap='gray')
            plt.title('Segmented Structure')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.plot(positions_smoothed, curvatures_smoothed, label=f'Segmented Structure {index + 1}')
            plt.title('Curvature as a Function of Position along Contour')
            plt.xlabel('Position along the contour')
            plt.ylabel('Curvature')
            plt.legend()
            plt.show()
            plt.close()

    return all_curvature_data



if __name__ == '__main__':
    combined_layer_full = np.load('combined_layer_full.npy')
    combined_layer_full = combined_layer_full.astype(np.uint16) # it is crucial to convert to uint explicitly,

    # uint16 of course means it can store up to 2^16 -1 = 65535 labels, which should be more than enough (2^8-1=255) not enough

    img_file_path = "/Users/px/Downloads/SDU x KU - Droplet & foam quantification/Images/Light microscopy/Emulsion droplets (light microscopy)/BF-Droplet (1).tiff"  # @param {type:"string"}
    img_to_seg = np.array(Image.open(img_file_path))

    #@markdown ###Label scale:
    lstat_scale = 1.0 #@param {type:"number"}

    #@markdown ###Label size:
    lstat_size = True #@param {type:"boolean"}

    #@markdown ###Label intensity:
    lstat_intensity = False #@param {type:"boolean"}

    #@markdown ###Label perimeter:
    lstat_perimeter = True #@param {type:"boolean"}

    #@markdown ###Label shape:
    lstat_shape = True #@param {type:"boolean"}

    #@markdown ###Label position:
    lstat_position = True #@param {type:"boolean"}

    #@markdown ###Label moments:
    lstat_moments = True #@param {type:"boolean"}

    pd.set_option('display.max_columns', None)

    randomize_labels = False #@param {type:"boolean"}
    if randomize_labels:
        combined_layer_full = func_randomize_labels(combined_layer_full)

    table = label_statistics(rgb2grey(img_to_seg), combined_layer_full, size=lstat_size, intensity=lstat_intensity,
                             perimeter=lstat_perimeter, shape=lstat_shape, position=lstat_position, moments=lstat_moments, lstat_scale=lstat_scale)
    # Save table to csv (Download promt)
    table.to_csv('label_statistics.csv')

    plottings_structures(combined_layer_full)
    plt.savefig('plot.svg', format='svg', bbox_inches='tight')
    plt.show()

    ### neareast neighbor based on contour
    nearest_neighbor_distances = calculate_nearest_neighbor(combined_layer_full)
    plot_structures_with_lines(combined_layer_full, nearest_neighbor_distances)
    plot_histogram_nearest_neighbor(nearest_neighbor_distances)

    # Export to CSV
    export_to_csv(nearest_neighbor_distances)

    ### neareast neighbor based on centroid
    nearest_neighbor_distances_centroid = calculate_nearest_neighbor_centroid(combined_layer_full)
    plot_structures_with_lines_centroid(combined_layer_full, nearest_neighbor_distances_centroid)
    plot_histogram_nearest_neighbor(nearest_neighbor_distances_centroid)

    # one alternative instead of calculating centroid again is to make use of the previous table generated by simpleITK
    # We checked that they generate same results.
    """
    nearest_neighbor_distances_centroid_w_table = calculate_nearest_neighbor_centroid_w_table(structures, table)
    plot_structures_with_lines_centroid(structures, nearest_neighbor_distances_centroid_w_table)
    plot_histogram_nearest_neighbor_centroid(nearest_neighbor_distances_centroid_w_table)
    """

    # Save the combined DataFrame to a single CSV file
    all_curvature_data = contour_analysis(combined_layer_full, plot=True)
    csv_filename = 'curvature_data.csv'
    all_curvature_data.to_csv(csv_filename, index=False)

    results = contour_analysis_unfold(combined_layer_full)
    csv_filename = 'curvature_data_unfold.csv'
    results_table = pd.DataFrame(results)
    results_table.to_csv(csv_filename, index=False)


    # Plot Delta_r as a function of theta for each label
    for result in results:
        theta = [point[1] for point in result['sampled_points']]
        delta_r_normalized = result['delta_r_normalized']

        # Create figure and subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Plot Delta_r vs theta
        axs[0].plot(theta[:-1], delta_r_normalized)  # Skip the last theta point for plotting Delta_r
        axs[0].set_title(f"Label {result['label']} - Normalized Delta_r vs Theta")
        axs[0].set_xlabel("Theta (radians)")
        axs[0].set_ylabel("Normalized Delta_r")
        axs[0].grid(True)

        # Plot the corresponding label image
        axs[1].imshow(combined_layer_full == result['label'], cmap='gray')
        axs[1].set_title(f"Label {result['label']} - Label Image")
        axs[1].axis('off')

        plt.show()
        plt.close(fig) # it is important to close after each iteration.
