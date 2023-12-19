
message(STATUS "enable LAMMPS plugin mode")
add_library(maceLammpsPlugin INTERFACE)

target_include_directories(maceLammpsPlugin INTERFACE ${LAMMPS_INCLUDE_DIRS})
find_package(LAMMPS REQUIRED)
message(STATUS "LAMMPS version: ${LAMMPS_VERSION}")
target_link_libraries(maceLammpsPlugin INTERFACE LAMMPS::lammps)

find_package(MPI REQUIRED)

# set(LAMMPS_MPI_INCLUDE_DIRS ${MPI_CXX_INCLUDE_DIRS})
message(STATUS "LAMMPS_INCLUDE_DIRS is set to ${LAMMPS_INCLUDE_DIRS}")
message(STATUS "LAMMPS_LIBRARIES is set to ${LAMMPS_LIBRARIES}")
# message(STATUS "LAMMPS_MPI_INCLUDE_DIRS is ${LAMMPS_MPI_INCLUDE_DIRS}")
message(STATUS "MPI_CXX_LIBRARIES
is ${MPI_CXX_LIBRARIES}")
message(STATUS "MPI_LIBRARIES
is ${MPI_LIBRARIES}")
target_link_libraries(maceLammpsPlugin INTERFACE MPI::MPI_CXX)

if (NOT LAMMPS_VERSION_NUMBER)
  set(LAMMPS_VERSION_NUMBER ${LAMMPS_VERSION} PARENT_SCOPE)
  set(LAMMPS_VERSION_NUMBER ${LAMMPS_VERSION})
  message(STATUS "Use LAMMPS_VERSION from find_package: ${LAMMPS_VERSION}")
else()
  set(LAMMPS_VERSION_NUMBER
      ${LAMMPS_VERSION_NUMBER}
      PARENT_SCOPE)
endif()

file(GLOB LMP_PLUGIN_SRC *.cpp ../*.cpp)
message(LMP_PLUGIN_SRC: ${LMP_PLUGIN_SRC})
message(LAMMPS_VERSION_NUMBER: ${LAMMPS_VERSION_NUMBER})

function(_add_lmp_plugin_variant variant_name prec_def)
  set(libname "mace_lmp${variant_name}")

  add_library(${libname} MODULE ${LMP_PLUGIN_SRC})

  message(STATUS "Use torch_libs: ${TORCH_LIBRARIES}")
  message(STATUS "Use torch_include_dirs: ${TORCH_INCLUDE_DIRS}")

  target_link_libraries(
    ${libname} PUBLIC maceLammpsPlugin ${TORCH_LIBRARIES})
  target_include_directories(
    ${libname}
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..
            ${LAMMPS_SOURCE_DIR}/src ${TORCH_INCLUDE_DIRS})

  set_target_properties(
    ${libname} PROPERTIES INSTALL_RPATH "$ORIGIN;${TORCH_LIBRARIES}"
                          LINK_FLAGS "-rdynamic")

  target_compile_definitions(
    ${libname}
    PUBLIC ${prec_def}
    PRIVATE "LMPPLUGIN" # fix header path
    PRIVATE "LAMMPS_VERSION_NUMBER=${LAMMPS_VERSION_NUMBER}")
  install(TARGETS ${libname} DESTINATION lib/)

  if(${LAMMPS_VERSION_NUMBER} GREATER_EQUAL 20220324)
    set(PLUGINNAME "mace_lammpsplugin.so")
    #lammps need the plugin named end with 'plugin'
    install(
      CODE "execute_process( \
      COMMAND ${CMAKE_COMMAND} -E make_directory \
	${CMAKE_INSTALL_PREFIX}/lib/${libname}/   \
      )")
    install(
      CODE "execute_process( \
      COMMAND ${CMAKE_COMMAND} -E create_symlink \
	../${CMAKE_SHARED_LIBRARY_PREFIX}${libname}${CMAKE_SHARED_LIBRARY_SUFFIX} \
      ${CMAKE_INSTALL_PREFIX}/lib/${libname}/${PLUGINNAME}   \
      )")
  endif()

endfunction()
_add_lmp_plugin_variant("high" "")
# _add_lmp_plugin_variant("${HIGH_PREC_VARIANT}" "${HIGH_PREC_DEF}")
# _add_lmp_plugin_variant("${LOW_PREC_VARIANT}" "${LOW_PREC_DEF}")

