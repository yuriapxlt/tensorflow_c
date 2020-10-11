package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tfc",
    visibility = ["//visibility:public"],
    srcs = [
        "tfc.cc",
    ],
    hdrs = [
        "tfc.h"
    ],
    deps = [
        "@opt_tensorflow//:tensorflow",
    ],    
)

exports_files(["session.pb"], visibility = ["//visibility:public"],)

filegroup(
    name = "session_pb",
    visibility = ["//visibility:public"],
    srcs = ["session.pb"]
)

cc_test(
    name = "test",
    visibility = ["//visibility:public"],
    srcs = ["test.cc"],
    deps = [
        "@opt_tensorflow//:tensorflow",
        "@com_google_googletest//:gtest_main",
        ":tfc",
    ],
    data = [
        ":session_pb",
    ],
)

cc_binary(
    name = "example",
    visibility = ["//visibility:public"],
    srcs = [
        "example.cc",
    ],
    copts = [],
    deps = [
        ":tfc"
    ],
    data = [
        ":session_pb",
    ],
)
