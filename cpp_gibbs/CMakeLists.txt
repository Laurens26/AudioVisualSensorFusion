cmake_minimum_required (VERSION 3.3)
project (lrfscpp)

# find_package (Eigen3 3.3 REQUIRED NO_MODULE)
# SET( EIGEN3_INCLUDE_DIR "C:/eigen-3.4.0" )
# IF( NOT EIGEN3_INCLUDE_DIR )
#     MESSAGE( FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
# ENDIF()

# Manually specify Eigen include directory
# SET(EIGEN3_INCLUDE_DIR "C:/Users/Laure/OneDrive/0_PAVKA/Technische_Unterlagen/0_Masterarbeit_SensorFusion_Laurens/Implementation/eigen-3.4.0")
SET(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../eigen-3.4.0")

IF(NOT EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Dense")
    MESSAGE(FATAL_ERROR "Eigen3 headers not found in ${EIGEN3_INCLUDE_DIR}. Please check the path.")
ENDIF()
INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )

# Manually specify EigenRand include directory
# SET(EIGENRAND_INCLUDE_DIR "C:/Users/Laure/OneDrive/0_PAVKA/Technische_Unterlagen/0_Masterarbeit_SensorFusion_Laurens/Implementation/EigenRand-0.5.1")
SET(EIGENRAND_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../EigenRand-0.5.1")
IF(NOT EXISTS "${EIGENRAND_INCLUDE_DIR}/EigenRand")
    MESSAGE(FATAL_ERROR "EIGENRAND_INCLUDE_DIR headers not found in ${EIGENRAND_INCLUDE_DIR}. Please check the path.")
ENDIF()
INCLUDE_DIRECTORIES ( "${EIGENRAND_INCLUDE_DIR}" )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")

add_subdirectory(pybind11)
pybind11_add_module(lrfscpp src/main.cpp)
# target_include_directories(lrfscpp PUBLIC C:/armadillo/include)
