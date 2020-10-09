cc_binary(
    name = "bla",
    srcs = ["main.cpp"],
    deps = [":tfc"],
)

cc_library(
    name = "tfc",
    srcs = ["tfc.cpp"],
    hdrs = ["tfc.h"],
)

cc_test(
    name = "bla_test",
    srcs = ["test.cpp"],
    deps = [
        "@com_google_googletest//:gtest_main",
        ":tfc",
    ],
)
