from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy.linalg as npl
import scipy.ndimage as ndi
from experiment_config import config
import pandas as pd

def _calculateAllPermutations(itemList):
    if len(itemList) == 1:
        return [[i] for i in itemList[0]]
    else:
        sub_permutations = _calculateAllPermutations(itemList[1:])
        return [[i] + p for i in itemList[0] for p in sub_permutations]

def worker_init_fn(worker_id):
    seed = int(torch.utils.data.get_worker_info().seed) % (2**32)
    np.random.seed(seed=seed)

def volumeTransform(
    image,
    voxel_spacing,
    transform_matrix,
    center=None,
    output_shape=None,
    output_voxel_spacing=None,
    **argv,
):

    if "offset" in argv:
        raise ValueError(
            "Cannot supply 'offset' to scipy.ndimage.affine_transform - already used by this function"
        )
    if "output_shape" in argv:
        raise ValueError(
            "Cannot supply 'output_shape' to scipy.ndimage.affine_transform - already used by this function"
        )

    if image.ndim != len(voxel_spacing):
        raise ValueError("Voxel spacing must have the same dimensions")

    if center is None:
        voxelCenter = (np.array(image.shape) - 1) / 2.0
    else:
        if len(center) != image.ndim:
            raise ValueError(
                "center point has not the same dimensionality as the image"
            )

        # Transform center to voxel coordinates
        voxelCenter = np.asarray(center) / voxel_spacing

    transform_matrix = np.asarray(transform_matrix)
    if output_voxel_spacing is None:
        if output_shape is None:
            output_voxel_spacing = np.ones(transform_matrix.shape[0])
        else:
            output_voxel_spacing = np.ones(len(output_shape))
    else:
        output_voxel_spacing = np.array(output_voxel_spacing)

    if transform_matrix.shape[1] != image.ndim:
        raise ValueError(
            "transform_matrix does not have the correct number of columns (does not match image dimensionality)"
        )
    if transform_matrix.shape[0] != image.ndim:
        raise ValueError(
            "Only allowing square transform matrices here, even though this is unneccessary. However, one will need an algorithm here to create full rank-square matrices. 'QR decomposition with Column Pivoting' would probably be a solution, but the author currently does not know what exactly this is, nor how to do this..."
        )

    # Normalize the transform matrix
    transform_matrix = np.array(transform_matrix)
    transform_matrix = (
        transform_matrix.T
        / np.sqrt(np.sum(transform_matrix * transform_matrix, axis=1))
    ).T

    transform_matrix = np.linalg.inv(
        transform_matrix.T
    )  # Important normalization for shearing matrices!!

    # The forwardMatrix transforms coordinates from input image space into result image space
    forward_matrix = np.dot(
        np.dot(np.diag(1.0 / output_voxel_spacing), transform_matrix),
        np.diag(voxel_spacing),
    )

    if output_shape is None:
        # No output dimensions are specified
        # Therefore we calculate the region that will span the whole image
        # considering the transform matrix and voxel spacing.
        image_axes = [[0 - o, x - 1 - o] for o, x in zip(voxelCenter, image.shape)]
        image_corners = _calculateAllPermutations(image_axes)

        transformed_image_corners = map(
            lambda x: np.dot(forward_matrix, x), image_corners
        )
        output_shape = [
            1 + int(np.ceil(2 * max(abs(x_max), abs(x_min))))
            for x_min, x_max in zip(
                np.amin(transformed_image_corners, axis=0),
                np.amax(transformed_image_corners, axis=0),
            )
        ]
    else:
        # Check output_shape
        if len(output_shape) != transform_matrix.shape[1]:
            raise ValueError(
                "output dimensions must match dimensionality of the transform matrix"
            )
    output_shape = np.array(output_shape)

    # Calculate the backwards matrix which will be used for the slice extraction
    backwards_matrix = npl.inv(forward_matrix)
    target_image_offset = voxelCenter - backwards_matrix.dot((output_shape - 1) / 2.0)

    return ndi.affine_transform(
        image,
        backwards_matrix,
        offset=target_image_offset,
        output_shape=output_shape,
        **argv,
    )

def clip_and_scale(npzarray, maxHU=400.0, minHU=-1000.0):
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1.0
    npzarray[npzarray < 0] = 0.0
    return npzarray

def rotateMatrixX(cosAngle, sinAngle):
    return np.asarray([[1, 0, 0], [0, cosAngle, -sinAngle], [0, sinAngle, cosAngle]])

def rotateMatrixY(cosAngle, sinAngle):
    return np.asarray([[cosAngle, 0, sinAngle], [0, 1, 0], [-sinAngle, 0, cosAngle]])

def rotateMatrixZ(cosAngle, sinAngle):
    return np.asarray([[cosAngle, -sinAngle, 0], [sinAngle, cosAngle, 0], [0, 0, 1]])




class CTCaseDataset(data.Dataset):
    def __init__(
        self,
        data_dir: str, 
        mask_data_dir: str,
        dataset: pd.DataFrame,
        translations: bool = None,
        rotations: tuple = None, 
        elastic: bool = None, 
        size_px: int = 64,
        size_mm: int = 50,
        mode: str = "2D",
    ):

        self.data_dir = Path(data_dir)
        self.mask_data_dir = Path(mask_data_dir)
        self.dataset = dataset
        self.patch_size = config.PATCH_SIZE
        self.rotations = rotations
        self.translations = translations
        self.elastic = elastic
        self.size_px = size_px
        self.size_mm = size_mm
        self.mode = mode


    def __getitem__(self, idx):  # caseid, z, y, x, label, radius

        pd = self.dataset.iloc[idx]

        label = pd.label

        annotation_id = pd.AnnotationID

        image_path = self.data_dir / "image" / f"{annotation_id}.npy"
        image_mask_path = self.mask_data_dir / "image" / f"{annotation_id}.npy" 
        metadata_path = self.data_dir / "metadata" / f"{annotation_id}.npy"

        
 

        # numpy memory map data/image case file
        img = np.load(image_path, mmap_mode="r")
        img_mask = np.load(image_mask_path, mmap_mode="r")
        metadata = np.load(metadata_path, allow_pickle=True).item()

        


        origin = metadata["origin"]
        spacing = metadata["spacing"]
        transform = metadata["transform"]

        translations = None
        if self.translations == True:
            radius = 2.5
            translations = radius if radius > 0 else None
        
        if self.mode == "2D":
            output_shape = (1, self.size_px, self.size_px)
        else:
            output_shape = (self.size_px, self.size_px, self.size_px)

        patch, mask = extract_patch(
            CTData=img, 
            mask=img_mask,
            coord=tuple(np.array(self.patch_size) // 2),
            srcVoxelOrigin=origin,
            srcWorldMatrix=transform,
            srcVoxelSpacing=spacing,
            output_shape=output_shape,
            voxel_spacing=(
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
            ),
            rotations=self.rotations,
            translations=translations,
            coord_space_world=False,
            mode=self.mode,
        )

        # ensure same datatype...
        patch = patch.astype(np.float32)
        mask = mask.astype(np.float32)

        # clip and scale...
        patch = clip_and_scale(patch)
        # mask = clip_and_scale(mask) # already its values be 0 to 1
        
        patch = torch.from_numpy(patch)
        mask = torch.from_numpy(mask)
        

        target = torch.ones((1,)) * label

        
        sample = {
            "image": patch, 
            "mask": mask,
            "label": target.long(),
            "ID": annotation_id,
        }

        return sample 
    # In case of 2D, shape : tensor(3, size_px, size_px)
    # In case of 3D, shape : tensor(1, size_px, size_px, size_px)

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        return fmt_str
    

def sample_random_coordinate_on_sphere(radius):
    # Generate three random numbers x,y,z using Gaussian distribution
    random_nums = np.random.normal(size=(3,))

    # You should handle what happens if x=y=z=0.
    if np.all(random_nums == 0):
        return np.zeros((3,))

    # Normalise numbers and multiply number by radius of sphere
    return random_nums / np.sqrt(np.sum(random_nums * random_nums)) * radius

def extract_patch(
    CTData, 
    mask, 
    coord,
    srcVoxelOrigin,
    srcWorldMatrix,
    srcVoxelSpacing,
    output_shape=(64, 64, 64),
    voxel_spacing=(50.0 / 64, 50.0 / 64, 50.0 / 64),
    rotations=None,
    translations=None, 
    coord_space_world=False,
    mode="2D",
):

    transform_matrix = np.eye(3)

    if rotations is not None:
        (zmin, zmax), (ymin, ymax), (xmin, xmax) = rotations

        # add random rotation
        angleX = np.multiply(np.pi / 180.0, np.random.randint(xmin, xmax, 1))[0]
        angleY = np.multiply(np.pi / 180.0, np.random.randint(ymin, ymax, 1))[0]
        angleZ = np.multiply(np.pi / 180.0, np.random.randint(zmin, zmax, 1))[0]

        transformMatrixAug = np.eye(3)
        transformMatrixAug = np.dot(
            transformMatrixAug, rotateMatrixX(np.cos(angleX), np.sin(angleX))
        )
        transformMatrixAug = np.dot(
            transformMatrixAug, rotateMatrixY(np.cos(angleY), np.sin(angleY))
        )
        transformMatrixAug = np.dot(
            transformMatrixAug, rotateMatrixZ(np.cos(angleZ), np.sin(angleZ))
        )

        transform_matrix = np.dot(transform_matrix, transformMatrixAug)

    if translations is not None:
        # add random translation
        radius = np.random.random_sample() * translations
        offset = sample_random_coordinate_on_sphere(radius=radius)
        offset = offset * (1.0 / srcVoxelSpacing)

        coord = np.array(coord) + offset

    # Normalize transform matrix
    thisTransformMatrix = transform_matrix

    thisTransformMatrix = (
        thisTransformMatrix.T
        / np.sqrt(np.sum(thisTransformMatrix * thisTransformMatrix, axis=1))
    ).T

    invSrcMatrix = np.linalg.inv(srcWorldMatrix)

    # world coord sampling
    if coord_space_world:
        overrideCoord = invSrcMatrix.dot(coord - srcVoxelOrigin)
    else:
        # image coord sampling
        overrideCoord = coord * srcVoxelSpacing
    overrideMatrix = (invSrcMatrix.dot(thisTransformMatrix.T) * srcVoxelSpacing).T

    patch = volumeTransform(
        CTData,
        srcVoxelSpacing,
        overrideMatrix,
        center=overrideCoord,
        output_shape=np.array(output_shape),
        output_voxel_spacing=np.array(voxel_spacing),
        order=1,
        prefilter=False,
    ) 
    
    mask_patch = volumeTransform(
        mask,
        srcVoxelSpacing,
        overrideMatrix,
        center=overrideCoord,
        output_shape=np.array(output_shape),
        output_voxel_spacing=np.array(voxel_spacing),
        order=1,
        prefilter=False,
    ) 

    if mode == "2D":
        # replicate the channel dimension
        patch = np.repeat(patch, 3, axis=0)
        mask_patch = np.repeat(mask_patch, 3, axis=0)

    else:
        patch = np.expand_dims(patch, axis=0)
        mask_patch = np.expand_dims(mask_patch, axis=0)

    return patch, mask_patch

def get_data_loader(
    data_dir, 
    mask_data_dir,
    dataset,
    mode="2D",
    sampler=None,
    workers=0,
    batch_size=64,
    size_px=64,
    size_mm=70,
    rotations=None,
    translations=None, 
):

    data_set = CTCaseDataset(
        data_dir=data_dir, 
        mask_data_dir=mask_data_dir,
        translations=translations,
        dataset=dataset,
        rotations=rotations,
        size_mm=size_mm,
        size_px=size_px,
        mode=mode,
    )

    shuffle = False
    # if sampler == None:
    #     shuffle = (True,)

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        sampler=sampler,
        worker_init_fn=worker_init_fn,
    )


    return data_loader



def test():
    # Test the dataloader
    import pandas as pd
    from experiment_config import config
    import matplotlib.pyplot as plt

    dataset = pd.read_csv(config.CSV_DIR_TRAIN)

    train_loader = get_data_loader(
        data_dir=config.DATADIR,
        dataset=dataset,
        mode=config.MODE,
        workers=8,
        batch_size=config.BATCH_SIZE,
        size_px=config.SIZE_PX,
        size_mm=config.SIZE_MM,
        rotations=config.ROTATION,
        translations=config.TRANSLATION, 
        elastic=config.ELASTIC
    )

    for i, data in enumerate(train_loader):
        print(i, data["image"].shape, data["label"].shape)

if __name__ == "__main__":
    test()
