load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

googletest_version = 'release-1.10.0'
http_archive(
     name = "com_google_googletest",
     urls = ["https://github.com/google/googletest/archive/{}.zip".format(googletest_version)],
     strip_prefix = "googletest-{}".format(googletest_version),
)
