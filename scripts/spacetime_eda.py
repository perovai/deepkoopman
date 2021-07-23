import sys
from pathlib import Path

from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from aiphysim.dataloading.dataloader_spacetime import RB2DataLoader  # noqa: E402

if __name__ == "__main__":

    folder = str(Path("~/Downloads").expanduser())
    file = "rb2d_ra1e6_s42.npz"

    ds = RB2DataLoader(data_dir=folder, data_filename=file)

    dl = DataLoader(ds, batch_size=7)

    dl_it = iter(dl)

    batch = next(dl_it)

    space_time_crop_lres, point_coord, point_value = batch

    print(
        space_time_crop_lres.shape,
        "space_time_crop_lres `batch x 4, nt_lres, nz_lres, nx_lres`",
        "where 4 are the phys channels pbuw",
    )

    print(
        point_coord.shape,
        "space_time_crop_lres: array of shape [4, nt_lres, nz_lres, nx_lres],",
        "where 4 are the phys channels pbuw.",
    )
    print(
        point_value.shape,
        "point_value: array of shape [n_samp_pts_per_crop, 4],",
        "where 4 are the phys channels pbuw.",
    )
