/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Modified from https://github.com/ACEsuit/lammps
   by:
      William C Witt (University of Cambridge)
------------------------------------------------------------------------- */

#include "pair_mace.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairMACE::PairMACE(LAMMPS *lmp) : Pair(lmp)
{
  no_virial_fdotr_compute = 1;
}

/* ---------------------------------------------------------------------- */

PairMACE::~PairMACE()
{
}

/* ---------------------------------------------------------------------- */

void PairMACE::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  if (atom->nlocal != list->inum) error->all(FLERR, "ERROR: nlocal != inum.");
  if (domain_decomposition) {
    if (atom->nghost != list->gnum) error->all(FLERR, "ERROR: nghost != gnum.");
  }

  // ----- positions -----
  int n_nodes;
  if (domain_decomposition) {
    n_nodes = atom->nlocal + atom->nghost;
  } else {
    // normally, ghost atoms are included in the graph as independent
    // nodes, as required when the local domain does not have PBC.
    // however, in no_domain_decomposition mode, ghost atoms are known to
    // be shifted versions of local atoms.
    n_nodes = atom->nlocal;
  }
  auto positions = torch::empty({n_nodes,3}, torch_float_dtype);
  #pragma omp parallel for
  for (int ii=0; ii<n_nodes; ++ii) {
    int i = list->ilist[ii];
    positions[i][0] = atom->x[i][0];
    positions[i][1] = atom->x[i][1];
    positions[i][2] = atom->x[i][2];
  }

  // ----- cell -----
  auto cell = torch::zeros({3,3}, torch_float_dtype);
  cell[0][0] = domain->h[0];
  cell[0][1] = 0.0;
  cell[0][2] = 0.0;
  cell[1][0] = domain->h[5];
  cell[1][1] = domain->h[1];
  cell[1][2] = 0.0;
  cell[2][0] = domain->h[4];
  cell[2][1] = domain->h[3];
  cell[2][2] = domain->h[2];

  // ----- edge_index and unit_shifts -----
  // count total number of edges
  int n_edges = 0;
  std::vector<int> n_edges_vec(n_nodes, 0);
  #pragma omp parallel for reduction(+:n_edges)
  for (int ii=0; ii<n_nodes; ++ii) {
    int i = list->ilist[ii];
    double xtmp = atom->x[i][0];
    double ytmp = atom->x[i][1];
    double ztmp = atom->x[i][2];
    int *jlist = list->firstneigh[i];
    int jnum = list->numneigh[i];
    for (int jj=0; jj<jnum; ++jj) {
      int j = jlist[jj];
      j &= NEIGHMASK;
      double delx = xtmp - atom->x[j][0];
      double dely = ytmp - atom->x[j][1];
      double delz = ztmp - atom->x[j][2];
      double rsq = delx * delx + dely * dely + delz * delz;
      if (rsq < r_max_squared) {
        n_edges += 1;
        n_edges_vec[ii] += 1;
      }
    }
  }
  // make first_edge vector to help with parallelizing following loop
  std::vector<int> first_edge(n_nodes);
  first_edge[0] = 0;
  for (int ii=0; ii<n_nodes-1; ++ii) {
    first_edge[ii+1] = first_edge[ii] + n_edges_vec[ii];
  }
  // fill edge_index and unit_shifts tensors
  auto edge_index = torch::empty({2,n_edges}, torch::dtype(torch::kInt64));
  auto unit_shifts = torch::zeros({n_edges,3}, torch_float_dtype);
  auto shifts = torch::zeros({n_edges,3}, torch_float_dtype);
  #pragma omp parallel for
  for (int ii=0; ii<n_nodes; ++ii) {
    int i = list->ilist[ii];
    double xtmp = atom->x[i][0];
    double ytmp = atom->x[i][1];
    double ztmp = atom->x[i][2];
    int *jlist = list->firstneigh[i];
    int jnum = list->numneigh[i];
    int k = first_edge[ii];
    for (int jj=0; jj<jnum; ++jj) {
      int j = jlist[jj];
      j &= NEIGHMASK;
      double delx = xtmp - atom->x[j][0];
      double dely = ytmp - atom->x[j][1];
      double delz = ztmp - atom->x[j][2];
      double rsq = delx * delx + dely * dely + delz * delz;
      if (rsq < r_max_squared) {
        edge_index[0][k] = i;
        if (domain_decomposition) {
          edge_index[1][k] = j;
        } else {
          int j_local = atom->map(atom->tag[j]);
          edge_index[1][k] = j_local;
          double shiftx = atom->x[j][0] - atom->x[j_local][0];
          double shifty = atom->x[j][1] - atom->x[j_local][1];
          double shiftz = atom->x[j][2] - atom->x[j_local][2];
          double shiftxs = std::round(domain->h_inv[0]*shiftx + domain->h_inv[5]*shifty + domain->h_inv[4]*shiftz);
          double shiftys = std::round(domain->h_inv[1]*shifty + domain->h_inv[3]*shiftz);
          double shiftzs = std::round(domain->h_inv[2]*shiftz);
          unit_shifts[k][0] = shiftxs;
          unit_shifts[k][1] = shiftys;
          unit_shifts[k][2] = shiftzs;
          shifts[k][0] = domain->h[0]*shiftxs + domain->h[5]*shiftys + domain->h[4]*shiftzs;
          shifts[k][1] = domain->h[1]*shiftys + domain->h[3]*shiftzs;
          shifts[k][2] = domain->h[2]*shiftzs;
        }
        k++;
      }
    }
  }

  // ----- node_attrs -----
  int n_node_feats = mace_atomic_numbers.size();
  auto node_attrs = torch::zeros({n_nodes,n_node_feats}, torch_float_dtype);
  #pragma omp parallel for
  for (int ii=0; ii<n_nodes; ++ii) {
    int i = list->ilist[ii];
    node_attrs[i][mace_type(atom->type[i])-1] = 1.0;
  }

  // ----- mask for ghost -----
  auto mask = torch::zeros(n_nodes, torch::dtype(torch::kBool));
  #pragma omp parallel for
  for (int ii=0; ii<atom->nlocal; ++ii) {
    int i = list->ilist[ii];
    mask[i] = true;
  }

  auto batch = torch::zeros({n_nodes}, torch::dtype(torch::kInt64));
  auto energy = torch::empty({1}, torch_float_dtype);
  auto forces = torch::empty({n_nodes,3}, torch_float_dtype);
  auto ptr = torch::empty({2}, torch::dtype(torch::kInt64));
  auto weight = torch::empty({1}, torch_float_dtype);
  ptr[0] = 0;
  ptr[1] = n_nodes;
  weight[0] = 1.0;

  // transfer data to device
  batch = batch.to(device);
  cell = cell.to(device);
  edge_index = edge_index.to(device);
  energy = energy.to(device);
  forces = forces.to(device);
  node_attrs = node_attrs.to(device);
  positions = positions.to(device);
  ptr = ptr.to(device);
  shifts = shifts.to(device);
  unit_shifts = unit_shifts.to(device);
  weight = weight.to(device);

  // pack the input, call the model
  c10::Dict<std::string, torch::Tensor> input;
  input.insert("batch", batch);
  input.insert("cell", cell);
  input.insert("edge_index", edge_index);
  input.insert("energy", energy);
  input.insert("forces", forces);
  input.insert("node_attrs", node_attrs);
  input.insert("positions", positions);
  input.insert("ptr", ptr);
  input.insert("shifts", shifts);
  input.insert("unit_shifts", unit_shifts);
  input.insert("weight", weight);
  auto output = model.forward({input, mask.to(device), bool(vflag_global)}).toGenericDict();

  // mace energy
  //   -> sum of site energies of local atoms
  if (eflag_global) {
    energy = output.at("total_energy_local").toTensor().cpu();
    eng_vdwl += energy.item<double>();
  }

  // mace forces
  //   -> derivatives of total mace energy
  forces = output.at("forces").toTensor().cpu();
  #pragma omp parallel for
  for (int ii=0; ii<n_nodes; ++ii) {
    int i = list->ilist[ii];
    atom->f[i][0] += forces[i][0].item<double>();
    atom->f[i][1] += forces[i][1].item<double>();
    atom->f[i][2] += forces[i][2].item<double>();
  }

  // mace site energies
  //   -> local atoms only
  if (eflag_atom) {
    auto node_energy = output.at("node_energy").toTensor().cpu();
    #pragma omp parallel for
    for (int ii=0; ii<list->inum; ++ii) {
      int i = list->ilist[ii];
      eatom[i] = node_energy[i].item<double>();
    }
  }

  // mace virials (local atoms only)
  //   -> derivatives of sum of site energies of local atoms
  if (vflag_global) {
    auto vir = output.at("virials").toTensor().cpu();
    virial[0] += vir[0][0][0].item<double>();
    virial[1] += vir[0][1][1].item<double>();
    virial[2] += vir[0][2][2].item<double>();
    virial[3] += 0.5*(vir[0][1][0].item<double>() + vir[0][0][1].item<double>());
    virial[4] += 0.5*(vir[0][2][0].item<double>() + vir[0][0][2].item<double>());
    virial[5] += 0.5*(vir[0][2][1].item<double>() + vir[0][1][2].item<double>());
  }

  // mace site virials
  //   -> not available
  if (vflag_atom) {
    error->all(FLERR, "ERROR: pair_mace does not support vflag_atom.");
  }
}

/* ---------------------------------------------------------------------- */

void PairMACE::settings(int narg, char **arg)
{
  if (narg > 1) {
    error->all(FLERR, "Too many pair_style arguments for pair_style mace.");
  }

  if (narg == 1) {
    if (strcmp(arg[0], "no_domain_decomposition") == 0) {
      domain_decomposition = false;
      // TODO: add check against MPI rank
    } else {
      error->all(FLERR, "Unrecognized argument for pair_style mace.");
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairMACE::coeff(int narg, char **arg)
{
  // TODO: remove print statements from this routine, or have a single proc print

  if (!allocated) allocate();

  if (!torch::cuda::is_available()) {
    std::cout << "CUDA unavailable, setting device type to torch::kCPU." << std::endl;
    device = c10::Device(torch::kCPU);
  } else {
    std::cout << "CUDA found, setting device type to torch::kCUDA." << std::endl;
    MPI_Comm local;
    MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
    int localrank;
    MPI_Comm_rank(local, &localrank);
    device = c10::Device(torch::kCUDA,localrank);
  }

  std::cout << "Loading MACE model from \"" << arg[2] << "\" ...";
  model = torch::jit::load(arg[2], device);
  std::cout << " finished." << std::endl;

  // extract default dtype from mace model
  for (auto p: model.named_attributes()) {
      // this is a somewhat random choice of variable to check. could it be improved?
      if (p.name == "model.node_embedding.linear.weight") {
          if (p.value.toTensor().dtype() == caffe2::TypeMeta::Make<float>()) {
            torch_float_dtype = torch::kFloat32;
          } else if (p.value.toTensor().dtype() == caffe2::TypeMeta::Make<double>()) {
            torch_float_dtype = torch::kFloat64;
          }
      }
  }
  std::cout << "  - The torch_float_dtype is: " << torch_float_dtype << std::endl;

  // extract r_max from mace model
  r_max = model.attr("r_max").toTensor().item<double>();
  r_max_squared = r_max*r_max;
  std::cout << "  - The r_max is: " << r_max << "." << std::endl;
  num_interactions = model.attr("num_interactions").toTensor().item<int64_t>();
  std::cout << "  - The model has: " << num_interactions << " layers." << std::endl;

  // extract atomic numbers from mace model
  auto a_n = model.attr("atomic_numbers").toTensor();
  for (int i=0; i<a_n.size(0); ++i) {
    mace_atomic_numbers.push_back(a_n[i].item<int64_t>());
  }
  std::cout << "  - The MACE model atomic numbers are: " << mace_atomic_numbers << "." << std::endl;

  // extract atomic numbers from pair_coeff
  for (int i=3; i<narg; ++i) {
    auto iter = std::find(periodic_table.begin(), periodic_table.end(), arg[i]);
    int index = std::distance(periodic_table.begin(), iter) + 1;
    lammps_atomic_numbers.push_back(index);
  }
  std::cout << "  - The pair_coeff atomic numbers are: " << lammps_atomic_numbers << "." << std::endl;

  for (int i=1; i<=lammps_atomic_numbers.size(); ++i) {
    std::cout << "  - Mapping LAMMPS type " << i
      << " (" << periodic_table[lammps_atomic_numbers[i-1]-1]
      << ") to MACE type " << mace_type(i) << "." << std::endl;
  }

  for (int i=1; i<atom->ntypes+1; i++)
    for (int j=i; j<atom->ntypes+1; j++)
      setflag[i][j] = 1;
}

void PairMACE::init_style()
{
  if (force->newton_pair == 0) error->all(FLERR, "ERROR: Pair style mace requires newton pair on.");

  /*
    MACE requires the full neighbor list AND neighbors of ghost atoms
    it appears that:
      * without REQ_GHOST
           list->gnum == 0
           list->ilist does not include ghost atoms, but the jlists do
      * with REQ_GHOST
           list->gnum == atom->nghost
           list->ilist includes ghost atoms
  */
  if (domain_decomposition) {
    neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
  } else {
    neighbor->add_request(this, NeighConst::REQ_FULL);
  }
}

double PairMACE::init_one(int i, int j)
{
  // to account for message passing, require cutoff of n_layers * r_max
  return num_interactions*model.attr("r_max").toTensor().item<double>();
}

void PairMACE::allocate()
{
  allocated = 1;

  memory->create(setflag, atom->ntypes+1, atom->ntypes+1, "pair:setflag");
  for (int i=1; i<atom->ntypes+1; i++)
    for (int j=i; j<atom->ntypes+1; j++)
      setflag[i][j] = 0;

  memory->create(cutsq, atom->ntypes+1, atom->ntypes+1, "pair:cutsq");
}

int PairMACE::mace_type(int lammps_type)
{
    for (int i=0; i<mace_atomic_numbers.size(); ++i) {
      if (mace_atomic_numbers[i]==lammps_atomic_numbers[lammps_type-1]) {
        return i+1;
      }
    }
    error->all(FLERR, "Problem converting lammps_type to mace_type.");
    return -1;
 }