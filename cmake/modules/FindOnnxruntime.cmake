find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  PATHS
  ${ONNXRUNTIME_ROOTDIR}/lib
)

find_path(ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  PATHS ${ONNXRUNTIME_ROOTDIR}/include
  )
if(ONNXRUNTIME_LIBRARY AND ONNXRUNTIME_INCLUDE_DIR)
  set(ONNXRUNTIME_FOUND TRUE)
else()
  set(ONNXRUNTIME_FOUND FALSE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Onnxruntime
  REQUIRED_VARS ONNXRUNTIME_LIBRARY
)

if(ONNXRUNTIME_FOUND)
  add_library(Onnxruntime::Onnxruntime SHARED IMPORTED)
  set_target_properties(Onnxruntime::Onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
  )
endif()
set(ONNXRUNTIME_FOUND TRUE CACHE BOOL "Set to TRUE if onnxruntime is found")
