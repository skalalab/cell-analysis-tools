`import tifffile
from pathlib import Path
from ipywidgets import Box
import ipywidgets as widgets
import scipy.ndimage as ndi
from tkinter import Tk
from tkinter.filedialog import askdirectory


base_path = '.'



def roi_filler_widget_driver():
    global base_path

    layout = widgets.Layout(width='auto', height='40px')  # set width and height

    directory_button = widgets.Button(
        description='Pick starting directory:',
        indent=False,
        disabled=False,
        display='flex',
        flex_flow='column',
        align_items='stretch',
        layout=layout
    )
    convert_files_button = widgets.Button(
        description='Convert Files!',
        indent=False,
        disabled=False,
        display='flex',
        flex_flow='column',
        align_items='stretch',
        layout=layout
    )
    keys = ['initial directory',
            'convert files']

    items = [directory_button,
             convert_files_button]

    base_path = directory_button.on_click(generate_file_dialog_initial)

    convert_files_button.on_click(convert_files)

    button_dict = {keys[i]: items[i] for i in range(len(keys))}
    box = Box(children=items)
    box.layout.display = 'flex'
    box.layout.flex_flow = 'column'
    box.layout.align_items = 'stretch'
    return button_dict, box


def generate_file_dialog_initial(a):
    global base_path
    root = Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    folder = askdirectory(parent=root)
    root.destroy()

    if folder != '':
        base_path = folder
        if a is not None:
            a.description = f'Starting directory: {folder}'
        print(f'Updated the initial directory: {folder}')
        return True
    else:
        print('Initial directory not changed.')
        return False


def convert_files(a):
    global base_path

    if base_path == '':
        base_path = '.'

    convert_to_filled_and_labeled_rois()


def fill_and_label_rois(curr_nuclei): 
	"""
	Fills and labels ROI outlines using unique ints for each region. 

	Args: 
		:param curr_nucelei: Current image to process
	Returns: 
		:return: ROIs filled and labeled with unique int representations
	"""
	return ndi.label(ndi.binary_fill_holes(curr_nuclei))[0]

def convert_to_filled_and_labeled_rois():
    """
    Convert every nuclei mask to a filled and labeled version.

    Args:
        :param basedir: Directory to search through for nuclei mask files
    Returns:
        :return: Converted_files: List of TIFF file names that were converted
    """
    global base_path

    if base_path is None:
        if generate_file_dialog_initial(None) is False:
            print('Please select base directory before proceeding!')
            return None
    filled_labeled_rois = list()
    export_dir = '/filled_labeled_rois/'
    export_path = Path(base_path + export_dir)
    export_path.mkdir(exist_ok=True)

    for nuclei_path in Path(base_path).rglob('*Nuclei.tif'):
        curr_nuclei = tifffile.imread(nuclei_path)[:, :, 0]
        curr_nuclei_filled = fill_and_label_rois(curr_nuclei)
        filled_labeled_rois.append(curr_nuclei_filled)
        curr_full_filename = Path(base_path + export_dir + nuclei_path.stem).with_suffix('.tiff')
        tifffile.imwrite(str(curr_full_filename), curr_nuclei_filled)
        print(f'Converted and exported {str(curr_full_filename)}')
    return filled_labeled_rois
