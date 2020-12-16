from collections import defaultdict
from collections.abc import Sized
from mpi4py import MPI
import numpy as np

from june.groups import ExternalSubgroup
from june.exc import SimulatorError

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()


class MovablePeople:
    """
    Holds information about people who might be present in a domain, but may or may not be be,
    given circumstances. They have skinny profiles, which only have their id, infection probability,
    susceptibility, home domain, and whether active or not. For now, we mimic the original structure,
    but with an additional interface.
    """

    def __init__(self):
        self.skinny_out = {}
        self.skinny_in = {}
        self.index = {}

    def add_person(self, person, external_subgroup):
        """ Add or update a person to the outward facing group """
        domain_id = external_subgroup.domain_id
        group_spec = external_subgroup.spec
        group_id = external_subgroup.group_id
        subgroup_type = external_subgroup.subgroup_type

        if domain_id not in self.skinny_out:
            self.skinny_out[domain_id] = {}  # allocate domain id
        if group_spec not in self.skinny_out[domain_id]:
            self.skinny_out[domain_id][group_spec] = {}
        if group_id not in self.skinny_out[domain_id][group_spec]:
            self.skinny_out[domain_id][group_spec][group_id] = {}
        if subgroup_type not in self.skinny_out[domain_id][group_spec][group_id]:
            self.skinny_out[domain_id][group_spec][group_id][subgroup_type] = {}

        if person.infected:
            view = [
                person.id,
                person.infection.transmission.probability,
                0.0,
                mpi_rank,
                True,
            ]
        else:
            view = [person.id, 0.0, person.susceptibility, mpi_rank, True]

        self.skinny_out[domain_id][group_spec][group_id][subgroup_type][
            person.id
        ] = view

    def delete_person(self, person, external_subgroup):
        """ Remove a person from the external subgroup. For now we actually do it. Later
        we may flag them. """
        domain_id = external_subgroup.domain_id
        group_spec = external_subgroup.spec
        group_id = external_subgroup.group_id
        subgroup_type = external_subgroup.subgroup_type
        try:
            del self.skinny_out[domain_id][group_spec][group_id][subgroup_type][
                person.id
            ]
            return 0
        except KeyError:
            return 1

    def serialise(self, rank):
        """ Hopefully more efficient than standard pickle"""
        keys, data = [], []
        if rank not in self.skinny_out:
            return None, None, 0
        for group_spec in self.skinny_out[rank]:
            for group_id in self.skinny_out[rank][group_spec]:
                for subgroup_type in self.skinny_out[rank][group_spec][group_id]:
                    keys.append(
                        (
                            group_spec,
                            group_id,
                            subgroup_type,
                            len(
                                self.skinny_out[rank][group_spec][group_id][
                                    subgroup_type
                                ]
                            ),
                        )
                    )
                    data += [
                        view
                        for pid, view in self.skinny_out[rank][group_spec][group_id][
                            subgroup_type
                        ].items()
                    ]
        outbound = np.array(data)
        return keys, outbound, outbound.shape[0]

    def update(self, rank, keys, rank_data):
        """ Update the information we have about people coming into our domain
        :param rank: domain of origin
        :param keys: dictionary keys for the group structure
        :param rank_data: numpy array of all the person data
        """
        index = 0

        for key in keys:
            group_spec, group_id, subgroup_type, n_data = key
            if group_spec not in self.skinny_in:
                self.skinny_in[group_spec] = {}
            if group_id not in self.skinny_in[group_spec]:
                self.skinny_in[group_spec][group_id] = {}
            if subgroup_type not in self.skinny_in[group_spec][group_id]:
                self.skinny_in[group_spec][group_id][subgroup_type] = {}
            data = rank_data[index : index + n_data]
            index += n_data

            try:
                self.skinny_in[group_spec][group_id][subgroup_type].update(
                    {
                        int(k): {"inf_prob": i, "susc": s, "dom": d, "active": a}
                        for k, i, s, d, a in data
                    }
                )
            except:
                print("failing", rank, "f-done")
                raise

    def combine_skinny_out_with_other(self, other: "MovablePeople"):
        update_nested(self.skinny_out, other.skinny_out)

def update_nested(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_nested(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def move_info(info2move):
    """
    Send a list of arrays of uint32 integers to all ranks,
    and receive arrays from all ranks.
    
    """
    # flatten list of uneven vectors of data, ensure correct type
    assert len(info2move) == mpi_size
    buffer = np.concatenate(info2move)
    assert buffer.dtype == np.uint32

    n_sending = len(buffer)
    count = np.array([len(x) for x in info2move])
    displ = np.array([sum(count[:p]) for p in range(len(info2move))])

    # send my count to all processes
    values = mpi_comm.alltoall(count)

    n_receiving = sum(values)

    # now all processes know how much data they will get,
    # and how much from each rank

    r_buffer = np.zeros(n_receiving, dtype=np.uint32)
    rdisp = np.array([sum(values[:p]) for p in range(len(values))])

    mpi_comm.Alltoallv(
        [buffer, count, displ, MPI.UINT32_T], [r_buffer, values, rdisp, MPI.UINT32_T]
    )

    return r_buffer, n_sending, n_receiving

def export_import_info(to_export, collate=False):
    """
    a fairly general but slow (?) way to send information between ranks. Send data per rank,
    recieve corrosponding information coming back.
    parameters
    ----------
    to_export
        information of the form of a dict {0: data_0, 1: data_1, ..., mpi_rank: data_i, ...}
        where i == (current) mpi_rank, data_i should have len() == 0, or be None (ie, don't need
        to . Raise exception if not.

    returns:
        collated_imports
            same format as to_export, but imported data.
    """
    reqs = []
    for rank in range(mpi_size):
        data = to_export[rank]
        if rank == mpi_rank:
            if isinstance(data, Sized):
                assert len(data) == 0
            else:
                assert data is None
            continue
        reqs.append(mpi_comm.isend(data, dest=rank, tag=600))

    to_import = {}
    for rank in range(mpi_size):
        if rank == mpi_rank:
            continue
        to_import[rank] = mpi_comm.recv(source=rank, tag=600)

    for r in reqs:
        r.wait()

    return to_import