package(default_visibility = ["//visibility:public"])

filegroup(
    name = "session_pb",
    visibility = ["//visibility:public"],
    srcs = ["session.pb"]
)

filegroup(
    name = "model_pb",
    visibility = ["//visibility:public"],
    srcs = ["model.pb"]
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
    data = [
        ":session_pb",
        ":model_pb",
    ],
)

cc_binary(
    name = "session",
    visibility = ["//visibility:public"],
    srcs = ["session.cc"],
    copts = [],
    deps = ["//tensorflow_c:api"],
    data = [":session_pb"],
)

cc_binary(
    name = "model",
    visibility = ["//visibility:public"],
    srcs = ["model.cc"],
    copts = [],
    deps = ["//tensorflow_c:api"],
    data = [":model_pb"],
)
