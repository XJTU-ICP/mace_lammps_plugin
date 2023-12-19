/**
* See https://docs.lammps.org/Developer_plugins.html
*/
#include "lammpsplugin.h"
#include "version.h"
#include "pair_mace.h"

using namespace LAMMPS_NS;

static Pair *pair_mace_creator(LAMMPS *lmp)
{
  return new PairMACE(lmp);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "mace";
  plugin.info = "LAMMPS interface for MACE";
  plugin.author = "mace team";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &pair_mace_creator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
