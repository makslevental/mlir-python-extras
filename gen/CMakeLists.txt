cmake_minimum_required(VERSION 3.20.0)
project(gen LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddMLIRPython)
include(HandleLLVMOptions)
include(CMakePrintHelpers)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_executable(Gen Gen.cpp)
target_link_libraries(Gen MLIRTableGenLib)

add_custom_target(GenTableGen DEPENDS Gen)
set(GEN_TABLEGEN_EXE Gen)

_flatten_mlir_python_targets(mlir_python_sources_deps MLIRPythonSources)

set(TD_FILES ${mlir_python_sources_deps})
set(OPS_GEN ${mlir_python_sources_deps})
list(FILTER TD_FILES INCLUDE REGEX python_bindings_td_file)
list(FILTER OPS_GEN INCLUDE REGEX ops_gen)

#message(WARNING "${TD_FILES}")
#message(WARNING "${OPS_GEN}")

function(my_tablegen ofn)
  tablegen(GEN ${ARGV})
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
    PARENT_SCOPE)

  # Get the current set of include paths for this td file.
  cmake_parse_arguments(ARG "" "" "DEPENDS;EXTRA_INCLUDES" ${ARGN})
  get_directory_property(tblgen_includes INCLUDE_DIRECTORIES)
  list(APPEND tblgen_includes ${ARG_EXTRA_INCLUDES})
  # Filter out any empty include items.
  list(REMOVE_ITEM tblgen_includes "")

  # Build the absolute path for the current input file.
  if (IS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
  else()
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR}/${LLVM_TARGET_DEFINITIONS})
  endif()

  # Append the includes used for this file to the tablegen_compile_commands
  # file.
  file(APPEND ${CMAKE_BINARY_DIR}/tablegen_compile_commands.yml
    "--- !FileInfo:\n"
    "  filepath: \"${LLVM_TARGET_DEFINITIONS_ABSOLUTE}\"\n"
    "  includes: \"${CMAKE_CURRENT_SOURCE_DIR};${tblgen_includes}\"\n"
    )
endfunction()

foreach(OPS_GEN_TARGET TD_FILE_TARGET IN ZIP_LISTS OPS_GEN TD_FILES)
  get_target_property(OPS_GEN_FILE "${OPS_GEN_TARGET}" INTERFACE_SOURCES)
  get_target_property(TD_FILE "${TD_FILE_TARGET}" INTERFACE_SOURCES)
#  cmake_print_variables(OPS_GEN_FILE TD_FILE)

  set(output_filename "python/${TD_FILE_TARGET}.py")
  set(LLVM_TARGET_DEFINITIONS ${TD_FILE})

  string(REPLACE "." ";" TD_QUALIFIER_LIST ${TD_FILE_TARGET})
  list(LENGTH TD_QUALIFIER_LIST TD_QUALIFIER_LIST_LENGTH)
  if(TD_QUALIFIER_LIST_LENGTH EQUAL 3)
    list(GET TD_QUALIFIER_LIST 0 DIALECT)
    list(GET TD_QUALIFIER_LIST 1 EXTENSION)
#    message(WARNING "${TD_FILE_TARGET}: DIALECT=${DIALECT} EXTENSION=${EXTENSION}")
    my_tablegen("${output_filename}" -gen-python-op-bindings
      -bind-dialect=${DIALECT}
      -dialect-extension=${EXTENSION})
  elseif(TD_QUALIFIER_LIST_LENGTH EQUAL 4)
    list(GET TD_QUALIFIER_LIST 2 DIALECT)
#    message(WARNING "${TD_FILE_TARGET}: DIALECT=${DIALECT}")
    my_tablegen("${output_filename}" -gen-python-op-bindings
      -bind-dialect=${DIALECT})
  endif()

  declare_mlir_python_sources("${TD_FILE_TARGET}.tablegen"
    ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}"
    SOURCES ${output_filename}
  )
endforeach()
