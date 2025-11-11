from typing import Dict, Tuple

from pathlib import Path
import json
from glob import glob
import SimpleITK
import numpy as np
from scipy.special import logit
import joblib
# from processor import MalignancyProcessor
from processor_ensemble import MalignancyProcessor


# INPUT_PATH = Path("./test/input")
# OUTPUT_PATH = Path("./test/output")
# RESOURCE_PATH = Path("./resources")

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("/opt/app/resources")


def transform(input_image, point):
    """

    Parameters
    ----------
    input_image: SimpleITK Image
    point: array of points

    Returns
    -------
    tNumpyOrigin

    """
    return np.array(
        list(
            reversed(
                input_image.TransformContinuousIndexToPhysicalPoint(
                    list(reversed(point))
                )
            )
        )
    )

def itk_image_to_numpy_image(input_image):
    """

    Parameters
    ----------
    input_image: SimpleITK image

    Returns
    -------
    numpyImage: SimpleITK image to numpy image
    header: dict containing origin, spacing and transform in numpy format

    """

    numpyImage = SimpleITK.GetArrayFromImage(input_image)
    numpyOrigin = np.array(list(reversed(input_image.GetOrigin())))
    numpySpacing = np.array(list(reversed(input_image.GetSpacing())))

    # get numpyTransform
    tNumpyOrigin = transform(input_image, np.zeros((numpyImage.ndim,)))
    tNumpyMatrixComponents = [None] * numpyImage.ndim
    for i in range(numpyImage.ndim):
        v = [0] * numpyImage.ndim
        v[i] = 1
        tNumpyMatrixComponents[i] = transform(input_image, v) - tNumpyOrigin
    numpyTransform = np.vstack(tNumpyMatrixComponents).dot(np.diag(1 / numpySpacing))

    # define necessary image metadata in header
    header = {
        "origin": numpyOrigin,
        "spacing": numpySpacing,
        "transform": numpyTransform,
    }

    return numpyImage, header


class NoduleProcessor:
    def __init__(self, ct_image_file, nodule_locations, clinical_information, mode="2D", model_name="LUNA25-baseline-2D"):
        """
        Parameters
        ----------
        ct_image_file: Path to the CT image file
        nodule_locations: Dictionary containing nodule coordinates and annotationIDs
        clinical_information: Dictionary containing clinical information (Age and Gender)
        mode: 2D or 3D
        model_name: Name of the model to be used for prediction
        """
        self._image_file = ct_image_file
        self.nodule_locations = nodule_locations
        self.clinical_information =clinical_information
        self.mode = mode
        self.model_name = model_name

        self.processor = MalignancyProcessor(mode=mode, suppress_logs=True, model_name=model_name)


    def predict(self, input_image: SimpleITK.Image, coords: np.array) -> Dict:
        """

        Parameters
        ----------
        input_image: SimpleITK Image
        coords: numpy array with list of nodule coordinates in /input/nodule-locations.json

        Returns
        -------
        malignancy risk of the nodules provided in /input/nodule-locations.json
        """

        numpyImage, header = itk_image_to_numpy_image(input_image)

        malignancy_risks = []
        for i in range(len(coords)):
            self.processor.define_inputs(numpyImage, header, [coords[i]])
            malignancy_risk, logits = self.processor.predict()
            malignancy_risk = np.array(malignancy_risk).reshape(-1)[0]
            malignancy_risks.append(malignancy_risk)

        malignancy_risks = np.array(malignancy_risks)


        malignancy_risks = list(malignancy_risks)

        return malignancy_risks

    def load_inputs(self):
        # load image
        print(f"Reading {self._image_file}")
        image = SimpleITK.ReadImage(str(self._image_file))

        self.annotationIDs = [p["name"] for p in self.nodule_locations["points"]]
        self.coords = np.array([p["point"] for p in self.nodule_locations["points"]])
        self.coords = np.flip(self.coords, axis=1)  # reverse to [z, y, x] format

        return image, self.coords, self.annotationIDs

    def process(self):
        """
        Load CT scan(s) and nodule coordinates, predict malignancy risk and write the outputs
        Returns
        -------
        None
        """
        image, coords, annotationIDs = self.load_inputs()
        output = self.predict(image, coords)

        assert len(output) == len(annotationIDs), "Number of outputs should match number of inputs"
        results = {
            "name": "Points of interest",
            "type": "Multiple points",
            "points": [],
            "version": {
                "major": 1,
                "minor": 0
            }
        }

        # Populate the "points" section dynamically
        coords = np.flip(coords, axis=1)
        for i in range(len(annotationIDs)):
            results["points"].append(
                    {
                    "name": annotationIDs[i],
                    "point": coords[i].tolist(),
                    "probability": float(output[i])
                    }
                )
        return results


def run(mode="2D", model_name="LUNA25-baseline-2D"):
    # Read the inputs
    input_nodule_locations = load_json_file(
        location=INPUT_PATH / "nodule-locations.json",
    )
    input_clinical_information = load_json_file(
        location=INPUT_PATH / "clinical-information-lung-ct.json",
    )
    input_chest_ct = load_image_path(
        location=INPUT_PATH / "images/chest-ct",
    )
    # # Read a resource file: the model weights
    # with open(RESOURCE_PATH / "some_resource.txt", "r") as f:
    #     print(f.read())
    
    # Validate access to GPU
    _show_torch_cuda_info()
    
    # Run your algorithm here
    processor = NoduleProcessor(ct_image_file=input_chest_ct,
                                nodule_locations=input_nodule_locations,
                                clinical_information=input_clinical_information,
                                mode=mode,
                                model_name=model_name)
    malignancy_risks = processor.process()

    # Save your output
    write_json_file(
        location=OUTPUT_PATH / "lung-nodule-malginancy-likelihoods.json",
        content=malignancy_risks,
    )
    print(f"Completed writing output to {OUTPUT_PATH}")
    print(f"Output: {malignancy_risks}") 
    return 0


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def load_image_path(*, location):
    # Use SimpleITK to read a file
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )

    assert (
                len(input_files) == 1
            ), "Please upload only one .mha file per job for grand-challenge.org"
    
    result = input_files[0]

    

    return result


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch version: {torch.version.cuda}")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    mode = "3D"
    model_name = "15epochs_5models"         # submission-ema_0.995-step30-flip_0.3-3D-20250720 
    raise SystemExit(run(mode= mode,
                         model_name=model_name))
