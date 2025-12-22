# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.22.1)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(mujoco_download-populate NONE)



include(ExternalProject)
ExternalProject_Add(mujoco_download-populate
                     "UPDATE_DISCONNECTED" "True" "URL" "https://github.com/google-deepmind/mujoco/releases/download/3.4.0/mujoco-3.4.0-linux-x86_64.tar.gz" "DOWNLOAD_EXTRACT_TIMESTAMP" "True"
                    SOURCE_DIR          "/home/yuan/arm_ws/src/mujoco_ros2_control/mujoco/mujoco-3.4.0"
                    BINARY_DIR          "/home/yuan/arm_ws/build/mujoco_ros2_control/_deps/mujoco_download-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
)


