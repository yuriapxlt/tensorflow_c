load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

googletest_version = 'release-1.10.0'
http_archive(
     name = "com_google_googletest",
     urls = ["https://github.com/google/googletest/archive/{}.zip".format(googletest_version)],
     strip_prefix = "googletest-{}".format(googletest_version),
     sha256 = "94c634d499558a76fa649edb13721dce6e98fb1e7018dfaeba3cd7a083945e91",
)

new_local_repository(
    name = "opt_tensorflow",
    path = "/opt/libtensorflow",
    build_file_content = 
"""
cc_library(
    name = "tensorflow",
    srcs = glob(["lib/**/*.so*"]),
    hdrs = glob(["include/**/*.h*"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
""",
)
