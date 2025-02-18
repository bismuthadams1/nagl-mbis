from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, as_completed
from openff.recharge.charges.resp import generate_resp_charge_parameter
from openff.recharge.charges.resp.solvers import IterativeSolver, SciPySolver
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.esp import ESPSettings
from openff.recharge.grids import GridGenerator, LatticeGridSettings
from openff.toolkit import Molecule
from openff.units import unit
import click
import tqdm
import gc
import shutil
import polars as pl
from polars import LazyFrame
import traceback

import numpy as np
from openff.units import unit
# from memory_profiler import profile

AU_ESP = unit.hartree / unit.e

# @ profile
class ESPCalculator:
    def __init__(self):
        self.ke = 1 / (4 * np.pi * unit.epsilon_0)  # 1/vacuum_permittivity, 1/(e**2 * a0 *Eh)

    def assign_esp(self, monopoles: np.ndarray, 
                   dipoles: np.ndarray, 
                   quadropules: np.ndarray, 
                   grid: unit.Quantity, 
                   coordinates: unit.Quantity) -> list:
        """Assign charges according to charge model selected

        Parameters
        ----------
        monopoles : np.ndarray
            Array of monopoles.
        dipoles : np.ndarray
            Array of dipoles.
        quadropules : np.ndarray
            Array of quadrupoles.
        grid : unit.Quantity
            Grid coordinates.
        coordinates : unit.Quantity
            Atom coordinates.

        Returns
        -------
        list of partial charges 
        """
        monopoles_quantity = monopoles * unit.e
        dipoles_quantity = dipoles * unit.e * unit.bohr
        quadropoles_quantity = quadropules * unit.e * unit.bohr * unit.bohr
        coordinates_ang = coordinates.to(unit.angstrom)

        monopole_esp = self._calculate_esp_monopole_au(grid_coordinates=grid, atom_coordinates=coordinates_ang, charges=monopoles_quantity)
        dipole_esp = self._calculate_esp_dipole_au(grid_coordinates=grid, atom_coordinates=coordinates_ang, dipoles=dipoles_quantity)
        quadrupole_esp = self._calculate_esp_quadropole_au(grid_coordinates=grid, atom_coordinates=coordinates_ang, quadrupoles=quadropoles_quantity)

        return (monopole_esp + dipole_esp + quadrupole_esp).m.flatten().tolist(), grid.m.tolist()

    def _calculate_esp_monopole_au(self, 
                                  grid_coordinates: unit.Quantity, 
                                  atom_coordinates: unit.Quantity, 
                                  charges: unit.Quantity) -> unit.Quantity:
        """Generate the esp from the on atom monopole

        Parameters
        ----------
        grid_coordinates : unit.Quantity
            Grid on which to build the esp on.
        atom_coordinates : unit.Quantity
            Coordinates of atoms to build the esp.
        charges : unit.Quantity
            Monopole or charges.

        Returns
        -------
        monopole_esp : unit.Quantity
            Monopole esp.
        """
        charges = charges.flatten()
        grid_coordinates = grid_coordinates.reshape((-1, 3)).to(unit.bohr)
        atom_coordinates = atom_coordinates.reshape((-1, 3)).to(unit.bohr)

        displacement = grid_coordinates[:, None, :] - atom_coordinates[None, :, :]
        distance = np.linalg.norm(displacement.m, axis=-1) * unit.bohr
        inv_distance = 1 / distance

        esp = self.ke * np.sum(inv_distance * charges[None, :], axis=1)
        return esp.to(AU_ESP)

    def _calculate_esp_dipole_au(self, 
                                grid_coordinates: unit.Quantity, 
                                atom_coordinates: unit.Quantity, 
                                dipoles: unit.Quantity) -> unit.Quantity:
        """Generate the esp from the on atom dipoles

        Parameters
        ----------
        grid_coordinates : unit.Quantity
            Grid on which to build the esp on.
        atom_coordinates : unit.Quantity
            Coordinates of atoms to build the esp.
        dipoles : unit.Quantity
            Dipoles or charges.

        Returns
        -------
        dipoles_esp : unit.Quantity
            Monopole esp.
        """
        dipoles = dipoles.to(unit.e * unit.bohr)
        grid_coordinates = grid_coordinates.reshape((-1, 3)).to(unit.bohr)
        atom_coordinates = atom_coordinates.reshape((-1, 3)).to(unit.bohr)

        displacement = grid_coordinates[:, None, :] - atom_coordinates[None, :, :]
        distance = np.linalg.norm(displacement.m, axis=-1) * unit.bohr
        inv_distance_cubed = 1 / distance**3
        dipole_dot = np.sum(displacement * dipoles[None, :, :], axis=-1)

        esp = self.ke * np.sum(inv_distance_cubed * dipole_dot, axis=1)
        return esp.to(unit.hartree / unit.e)

    def _calculate_esp_quadropole_au(self, 
                                    grid_coordinates: unit.Quantity, 
                                    atom_coordinates: unit.Quantity, 
                                    quadrupoles: unit.Quantity) -> unit.Quantity:
        """Generate the esp from the on atom quadrupoles

        Parameters
        ----------
        grid_coordinates : unit.Quantity
            Grid on which to build the esp on.
        atom_coordinates : unit.Quantity
            Coordinates of atoms to build the esp.
        quadrupoles : unit.Quantity
            Quadrupoles or charges.

        Returns
        -------
        quadrupoles_esp : unit.Quantity
            Monopole esp.
        """
        quadrupoles = self._detrace(quadrupoles.to(unit.e * unit.bohr * unit.bohr))
        grid_coordinates = grid_coordinates.reshape((-1, 3)).to(unit.bohr)
        atom_coordinates = atom_coordinates.reshape((-1, 3)).to(unit.bohr)

        displacement = grid_coordinates[:, None, :] - atom_coordinates[None, :, :]
        distance = np.linalg.norm(displacement.m, axis=-1) * unit.bohr
        inv_distance = 1 / distance

        quadrupole_dot_1 = np.sum(quadrupoles[None, :, :] * displacement[:, :, None], axis=-1)
        quadrupole_dot_2 = np.sum(quadrupole_dot_1 * displacement, axis=-1)
        esp = self.ke * np.sum((3 * quadrupole_dot_2 * (1 / 2 * inv_distance**5)), axis=-1)

        return esp.to(AU_ESP)

    def _detrace(self, quadrupoles: unit.Quantity) -> unit.Quantity:
        """Make sure we have the traceless quadrupole tensor.

        Parameters
        ----------
        quadrupoles : unit.Quantity
            Quadrupoles.

        Returns
        -------
        unit.Quantity
            Detraced quadrupoles.
        """
        quadrupoles = quadrupoles.m
        for i in range(quadrupoles.shape[0]):
            trace = np.trace(quadrupoles[i])
            trace /= 3
            quadrupoles[i][0][0] -= trace
            quadrupoles[i][1][1] -= trace
            quadrupoles[i][2][2] -= trace

        return quadrupoles * unit.e * unit.bohr * unit.bohr



def calculate_resp(table_entry: dict, grid_settings: LatticeGridSettings) -> dict:
    """
    For the entry in the pyarrow dataset calculate the resp charges and return the inverse distance and esp refernce grid.
    """
    qc_data_settings = ESPSettings(
        method="wb97x-d", basis="def2-tzvpp", grid_settings=grid_settings
    )
    off_molecule = Molecule.from_mapped_smiles(table_entry["smiles"], allow_undefined_stereo=True)
    conformer = (table_entry["conformation"] * unit.bohr).reshape(-1, 3)
    esp_grid = GridGenerator.generate(molecule=off_molecule, conformer=conformer, settings=grid_settings)
    dipoles = np.array(table_entry["mbis-dipoles"]).reshape(-1, 3)
    quads = np.array(table_entry["mbis-quadrupoles"]).reshape(-1, 3, 3)
    ref_esp, _ = ESPCalculator().assign_esp(
        monopoles=table_entry["mbis-charges"], 
        dipoles=dipoles, 
        quadropules=quads,
        grid=esp_grid,
        coordinates=conformer
        )
    ref_esp = (ref_esp * unit.hartree / unit.e).reshape(-1, 1)
    # use this esp to fid some RESP charges
    resp_solver = SciPySolver()
    qc_data_record = MoleculeESPRecord.from_molecule(
        molecule=off_molecule, 
        conformer=conformer,
        grid_coordinates=esp_grid,
        esp=ref_esp,
        esp_settings=qc_data_settings,
        electric_field=None
    )
    esp_charge_parameter = generate_resp_charge_parameter(
        [qc_data_record], resp_solver
    )
    # extract the charges into an array
    # we get back the tagged smiles with duplicated index, where the index matches the value in the arrary of the charge to use
    matchs = off_molecule.chemical_environment_matches(query=esp_charge_parameter.smiles)
    resp_charges = [0.0 for _ in range(off_molecule.n_atoms)]
    for match in matchs:
        for i, atom_indx in enumerate(match):
            resp_charges[atom_indx] = esp_charge_parameter.value[i]
    # create a new dict which can go into the new pyarrow table
    del esp_charge_parameter, resp_solver
    table_entry["resp-charges"] = resp_charges
    table_entry["esp"] = ref_esp.m.flatten().tolist()
    table_entry["esp_length"] = len(table_entry["esp"])
    displacement = (esp_grid[:, None, :] - conformer[None, :, :]).to(unit.bohr).m
    distance = np.linalg.norm(displacement, axis=-1) * unit.bohr
    inv_distance = 1 / distance
    table_entry["inv_distance"] = inv_distance.m.flatten().tolist()
    return table_entry

def iter_slices(df: LazyFrame, offset: int, batch_size: int) -> LazyFrame:
    row_count = df.select(pl.len()).collect().item()
    i = 0
    for offset in range(0, row_count, batch_size):
        i += 1
        yield i, df.slice(offset, batch_size).collect()



@click.command()
@click.option("-t", "--table", help="The name of the table to add the resp charges to")
@click.option("-o", "--output", help="The name of the new pyarrow table the resp charges should be saved to.")
@click.option("-p", "--processors", help="The number of processors to use while generating the resp charges.", default=2)
@click.option("-b", "--batch-size", help="The size of the batch to process.", default=1000)
def main(table: str, output: str, processors: int, batch_size: int):
    """
    Add the resp charges, ref esp and inverse grid distances to a pyarrow table and write to a new location
    """
    import os
    import shutil
    #scanning reduces memory overhead
    df = pl.scan_parquet(table)
    settings = LatticeGridSettings()
    os.makedirs("cache")
    offset = 0
    for i, bacth in iter_slices(df, offset, batch_size):
        offset +=batch_size
        with ProcessPoolExecutor(max_workers=processors, mp_context=get_context("spawn")) as pool:
            jobs = [
                pool.submit(calculate_resp, row, settings) for row in bacth.to_dicts()
            ]
            batch_results = []
            for result in tqdm.tqdm(as_completed(jobs), desc=f"Processing chunk {i}", ncols=80, total=len(jobs)):
                try:
                    batch_results.append(result.result())
                except Exception as e:
                    print(f"calc failed due to {e}")
                    print('printing traceback:')
                    traceback.print_exc()  # This will print the full traceback
                    print(traceback.format_exc())  # This will print the full traceback as a string
            # cache to file
            processed_df = pl.DataFrame(batch_results)
            # print(processed_df)
            output_file = os.path.join("cache", f"chunk_{i}.parquet")
            processed_df.write_parquet(output_file)
        # close the pool and del memory
        del processed_df
        gc.collect()


    # collect all the cache files
    processed_files = [os.path.join("cache", f) for f in os.listdir("cache") if f.endswith('.parquet')]
    combined_df = pl.concat([pl.scan_parquet(file) for file in processed_files])
    combined_df.sink_parquet(output)
    # remove the cache 

    shutil.rmtree("cache")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        shutil.rmtree("cache")
