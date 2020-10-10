cc_binary(
    name = "bla",
    srcs = ["bla.cc"],
    deps = [":tfc"],
)

cc_library(
    name = "tfc",
    srcs = ["tfc.cc"],
    hdrs = ["tfc.h"],
)

cc_test(
    name = "bla_test",
    srcs = ["bla_test.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        ":tfc",
    ],
)
