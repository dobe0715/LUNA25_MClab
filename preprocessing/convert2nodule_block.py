import SimpleITK as sitk
import numpy as np
import pandas
import multiprocessing
from pathlib import Path
import utils
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)

class NodulePreProcessor:
    """
    Class to preprocess nodule blocks and store as numpy files
    """

    def __init__(
        self,
        data_path: Path,
        csv_path: Path,
        save_path: Path,
    ) -> None:

        self.data_path = data_path
        self.save_path = save_path

        self.dataset = pandas.read_csv(csv_path)
        self.dataset = utils.standardize_nodule_df(self.dataset)

        self.dst_image_path = self.save_path / "image"
        self.dst_metadata_path = self.save_path / "metadata"

        self.dst_image_path.mkdir(exist_ok=True, parents=True)
        self.dst_metadata_path.mkdir(exist_ok=True, parents=True)


    def prepare_numpy_files(self, annotation_id):
        """Function to load a nifty file (prepared for nnU-Net)
        and convert that into numpy files for fast loading during training
        Args:
            annotation_id (str): The unique annotation ID for a nodule,
            f"{PatientID}_{LesionID}_{StudyDate}"
        """

        logging.info(f"processing {annotation_id}")

        image_path = self.data_path / f"{annotation_id}_0000.nii.gz"

        if image_path.is_file():

            image = sitk.ReadImage(str(image_path))
            image, header = utils.itk_image_to_numpy_image(image)

            np.save(self.dst_image_path / f"{annotation_id}.npy", image)
            np.save(self.dst_metadata_path / f"{annotation_id}.npy", header)

        else:
            logging.info(f"{annotation_id} missing")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Python script to convert nodule blocks from nifty files into numpy files"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        help="Path for CSV file containing annotations (this must contain PatientID, LesionID, StudyDate)",
        required=True,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Path to save the nodule blocks as numpy files",
        required=True,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to read the nifty files of nodule blocks",
        required=True,
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Number of workers for parallelization",
        required=False,
    )

    args = parser.parse_args()

    preprocessor = NodulePreProcessor(
        data_path=Path(args.data_path),
        csv_path=Path(args.csv_path),
        save_path=Path(args.save_path),
    )

    tasks = preprocessor.dataset.AnnotationID.unique()

    pool = multiprocessing.Pool(args.num_workers)
    pool.map(preprocessor.prepare_numpy_files, tasks)
    pool.close()