cmake_minimum_required(VERSION 3.16)
project(nomsbc C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# --- Options ---
option(NOMSBC_BUILD_DATAPREP "Build data preparation tool (requires libsbc)" ON)

# --- Dependencies ---

# kiss_fft: lightweight, embeddable FFT (BSD license)
include(FetchContent)
FetchContent_Declare(
    kiss_fft
    GIT_REPOSITORY https://github.com/mborgerding/kissfft.git
    GIT_TAG        131.1.0
)
FetchContent_MakeAvailable(kiss_fft)

# libsbc (optional, for data preparation only)
if(NOMSBC_BUILD_DATAPREP)
    find_library(SBC_LIB sbc)
    find_path(SBC_INCLUDE sbc/sbc.h)
    if(NOT SBC_LIB OR NOT SBC_INCLUDE)
        message(WARNING "libsbc not found -- data preparation tool will not be built. "
                        "Install with: apt install libsbc-dev")
        set(NOMSBC_BUILD_DATAPREP OFF)
    endif()
endif()

# libsndfile for reading/writing audio files
find_library(SNDFILE_LIB sndfile)
find_path(SNDFILE_INCLUDE sndfile.h)

# --- DSP library ---
add_library(nomsbc_dsp STATIC
    src/dsp/fft_wrap.c
    src/dsp/pitch.c
    src/dsp/cepstral.c
    src/dsp/features.c
)
target_include_directories(nomsbc_dsp PUBLIC src)
target_link_libraries(nomsbc_dsp PUBLIC kissfft::kissfft m)

# --- Enhancement modules library ---
add_library(nomsbc_modules STATIC
    src/modules/adacomb.c
    src/modules/adaconv.c
    src/modules/adashape.c
    src/modules/bwe.c
)
target_include_directories(nomsbc_modules PUBLIC src)
target_link_libraries(nomsbc_modules PUBLIC nomsbc_dsp)

# --- Enhancement pipeline ---
add_library(nomsbc_pipeline STATIC
    src/pipeline/enhance.c
)
target_include_directories(nomsbc_pipeline PUBLIC src)
target_link_libraries(nomsbc_pipeline PUBLIC nomsbc_modules)

# --- Data preparation tool ---
if(NOMSBC_BUILD_DATAPREP)
    add_library(nomsbc_sbc STATIC src/data/sbc_codec.c)
    target_include_directories(nomsbc_sbc PUBLIC src ${SBC_INCLUDE})
    target_link_libraries(nomsbc_sbc PUBLIC ${SBC_LIB})

    add_executable(prep_data src/data/prep_data.c)
    target_link_libraries(prep_data PRIVATE nomsbc_sbc nomsbc_dsp)
    if(SNDFILE_LIB AND SNDFILE_INCLUDE)
        target_compile_definitions(prep_data PRIVATE HAVE_SNDFILE)
        target_include_directories(prep_data PRIVATE ${SNDFILE_INCLUDE})
        target_link_libraries(prep_data PRIVATE ${SNDFILE_LIB})
    endif()
endif()

# --- Simple test / demo binary ---
add_executable(nomsbc_demo src/pipeline/demo.c)
target_link_libraries(nomsbc_demo PRIVATE nomsbc_pipeline)

# --- Darwin speech-enhanced audio tap ---
if(APPLE)
    add_library(nomsbc_darwin_core STATIC
        src/darwin/weights.c
        src/darwin/dnn_infer.c
        src/darwin/speech_detect.c
        src/darwin/audio_tap.c
    )
    target_include_directories(nomsbc_darwin_core PUBLIC src)
    target_link_libraries(nomsbc_darwin_core PUBLIC
        nomsbc_pipeline
        "-framework CoreAudio"
        "-framework AudioToolbox"
        "-framework CoreFoundation"
    )

    add_executable(nomsbc_darwin src/darwin/main.c)
    target_link_libraries(nomsbc_darwin PRIVATE nomsbc_darwin_core)
endif()
