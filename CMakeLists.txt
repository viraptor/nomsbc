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
set(KISSFFT_TEST OFF CACHE BOOL "" FORCE)
set(KISSFFT_TOOLS OFF CACHE BOOL "" FORCE)
set(KISSFFT_STATIC ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(kiss_fft)

# libsbc (vendored: google/libsbc in ./libsbc)
if(NOMSBC_BUILD_DATAPREP)
    set(LIBSBC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libsbc)
    if(NOT EXISTS ${LIBSBC_DIR}/include/sbc.h)
        message(WARNING "libsbc sources not found at ${LIBSBC_DIR} -- "
                        "data preparation tool will not be built.")
        set(NOMSBC_BUILD_DATAPREP OFF)
    else()
        add_library(sbc STATIC
            ${LIBSBC_DIR}/src/sbc.c
            ${LIBSBC_DIR}/src/bits.c
        )
        target_include_directories(sbc PUBLIC ${LIBSBC_DIR}/include)
        target_include_directories(sbc PRIVATE ${LIBSBC_DIR}/src)
    endif()
endif()

# libsndfile for reading/writing audio files (via pkg-config)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED IMPORTED_TARGET sndfile)
pkg_check_modules(SAMPLERATE REQUIRED IMPORTED_TARGET samplerate)

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
    target_include_directories(nomsbc_sbc PUBLIC src)
    target_link_libraries(nomsbc_sbc PUBLIC sbc)

    add_executable(prep_data src/data/prep_data.c)
    target_link_libraries(prep_data PRIVATE nomsbc_sbc nomsbc_dsp)
    target_compile_definitions(prep_data PRIVATE HAVE_SNDFILE)
    target_link_libraries(prep_data PRIVATE PkgConfig::SNDFILE PkgConfig::SAMPLERATE)
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
