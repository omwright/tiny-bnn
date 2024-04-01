from matplotlib import rc
from matplotlib.font_manager import fontManager
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import os

def hex_to_rgb(hex_string, normalize=True):
    """Convert a hex string to a normalized RGB color tuple."""
    hex_string = hex_string.lstrip('#')
    if normalize:   # Tuple of [0.0,1.0]
        rgb = tuple(int(hex_string[i:i+2], 16)/255.0 for i in (0, 2, 4))
    else:           # Tuple of [0,255]
        rgb = tuple(int(hex_string[i:i+2], 16) for i in (0, 2, 4))
    return rgb

def custom_colormap(color1, color2):
    """
    Creates a matplotlib colormap that transitions between two RGB colors.

    Args:
        color1 (tuple): A tuple of three floats between 0 and 1, representing an RGB color.
        color2 (tuple): A tuple of three floats between 0 and 1, representing an RGB color.
    
    Returns:
        A LinearSegmentedColormap object.
    """
    pos = [0.0, 1.0]
    colors = [(pos[0], color1), (pos[1], color2)]
    cmap = LinearSegmentedColormap('custom_colormap', colors)
    return cmap

def custom_colormap3(color1, color2, color3, pos=[0.0, 0.5, 1.0]):
    """
    A version with 3 colors.
    """
    colors = [(pos[0], color1), (pos[1], color2), (pos[2], color3)]
    cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)
    return cmap

def create_gif(directory, outpath, im_type='.png', duration=100, loop=0):
    """
    Create a GIF animation from images in the given directory.

    Args:
        directory: Path to the directory containing PNG images.
        outpath: Path to save the resulting GIF animation.
        im_type: '.png', '.jpg', etc.
        duration: Duration (in milliseconds) for each frame.
        loop: Number of loops. 0 means infinite loop.
    """
    im_files = [file for file in os.listdir(directory) if file.endswith(im_type)]
    im_files.sort()
    images = []
    for file in im_files:
        img = Image.open(os.path.join(directory, file))
        images.append(img)
    images[0].save(outpath, save_all=True, append_images=images[1:], duration=duration, loop=loop)

carnegie = hex_to_rgb('#C41230')
iron_gray = hex_to_rgb('#6D6E71')
steel_gray = hex_to_rgb('#E0E0E0')
scots_rose = hex_to_rgb('#EF3A47')
gold_thread = hex_to_rgb('#FDB515')
green_thread = hex_to_rgb('#009647')
teal_thread = hex_to_rgb('#008F91')
blue_thread = hex_to_rgb('#043673')
highland_sky = hex_to_rgb('#007BC0')

font_list = {f.name for f in fontManager.ttflist}
serif_fonts = ['Times New Roman', 'DejaVu Serif', 'serif']
for font in serif_fonts:
    if font in font_list:
        break
rc('font', **{'family': 'serif', 'serif': [font]})
rc('mathtext', fontset='stix')
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8