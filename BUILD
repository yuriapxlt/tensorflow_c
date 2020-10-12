package(default_visibility = ["//visibility:public"])

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
        "//tensorflow_c:api",
    ],
    data = [":session_pb"],
)

cc_binary(
    name = "example",
    visibility = ["//visibility:public"],
    srcs = ["example.cc"],
    copts = [],
    deps = ["//tensorflow_c:api"],
    data = [":session_pb"],
)
