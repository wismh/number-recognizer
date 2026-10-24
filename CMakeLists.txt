cmake_minimum_required(VERSION 3.20)
project(number-recognizer LANGUAGES CXX)

# The Wind SDK. The editor passes CMAKE_PREFIX_PATH when it builds the game; CMakeUserPresets.json does for an IDE.
find_package(Wind REQUIRED)

engine_add_game(number-recognizer
    src/main.cpp
    src/game.h
    src/game.cpp
    src/help/help.h
    src/help/help.cpp
    src/help/help_figures.h
    src/help/help_figures.cpp
    src/help/help_topic_table.h
    src/help/help_view_model.h
    src/loc/tr.h
    src/loc/locale_prefs.h
    src/loc/locale_prefs.cpp
    src/status_line.h
    src/recognizer_texts.h
    src/recognizer_texts.cpp
    src/dataset_manager.h
    src/dataset_manager.cpp
    src/neuron_panel.h
    src/neuron_panel.cpp
    src/recognizer.h
    src/recognizer.cpp
    src/model/digits.h
    src/model/parallel.h
    src/model/mlp.h
    src/model/mlp.cpp
    src/model/augment.h
    src/model/augment.cpp
    src/model/dream.h
    src/model/dream.cpp
    src/model/neuron_dreamer.h
    src/model/neuron_dreamer.cpp
    src/model/weight_picture.h
    src/model/weight_picture.cpp
    src/model/evaluation.h
    src/model/evaluation.cpp
    src/model/training_run.h
    src/model/training_run.cpp
    src/model/labeled_images.h
    src/model/dataset_store.h
    src/model/dataset_store.cpp
    src/model/normalize.h
    src/model/normalize.cpp
    src/model/mnist_data.h
    src/model/mnist_data.cpp
    src/model/model_store.h
    src/model/model_store.cpp
    src/view/digit_canvas.h
    src/view/digit_canvas.cpp
    src/view/list_item.h
    src/view/grid_row.h
    src/view/grid_row.cpp
    src/view/influence_paint.h
    src/view/influence_paint.cpp
    src/view/neuron_row.h
    src/view/neuron_row.cpp
    src/view/picture_paint.h
    src/view/picture_paint.cpp
    src/view/probability_paint.h
    src/view/probability_paint.cpp
    src/view/prototype_paint.h
    src/view/prototype_paint.cpp
    src/view/recognizer_view_model.h
    src/view/recognizer_view_model.cpp)

target_include_directories(number-recognizer PRIVATE src)

# MNIST IDX files are read from disk, not through the asset pipeline: copy them into assets/mnist beside the build
# output, which is also what the editor's Export packs.
add_custom_command(TARGET number-recognizer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/data/mnist"
        "$<TARGET_FILE_DIR:number-recognizer>/assets/mnist"
    COMMENT "Copy MNIST data beside number-recognizer"
    VERBATIM)
