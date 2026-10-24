# Unit tests of the domain (src/model). They link the same recognizer_model library as the game module and GTest from
# the SDK; no engine code is compiled or linked.
add_executable(recognizer_tests
    test_support.h
    test_augment.cpp
    test_dataset_store.cpp
    test_evaluation.cpp
    test_mlp.cpp
    test_mnist_data.cpp
    test_model_store.cpp
    test_normalize.cpp
    test_parallel.cpp
    test_training_run.cpp)
target_link_libraries(recognizer_tests PRIVATE recognizer_model GTest::gtest_main)
if(MSVC)
    if(WIND_SDK_CONFIG STREQUAL "Debug")
        set_target_properties(recognizer_tests PROPERTIES MSVC_RUNTIME_LIBRARY MultiThreadedDebugDLL)
    else()
        set_target_properties(recognizer_tests PROPERTIES MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
    endif()
endif()
gtest_discover_tests(recognizer_tests)
