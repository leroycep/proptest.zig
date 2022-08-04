const std = @import("std");

pub fn build(b: *std.build.Builder) void {
    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    const mode = b.standardReleaseOptions();

    const lib = b.addStaticLibrary("ofx-zig", "proptest.zig");
    lib.setBuildMode(mode);
    lib.install();

    const main_tests = b.addTest("proptest.zig");
    main_tests.setBuildMode(mode);

    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&main_tests.step);

    // Examples
    const sum = b.addTest("examples/00_sum.zig");
    sum.setBuildMode(mode);
    sum.addPackagePath("proptest", "./proptest.zig");

    const hello_tests = b.addTest("examples/01_hello.zig");
    hello_tests.setBuildMode(mode);
    hello_tests.addPackagePath("proptest", "./proptest.zig");

    const examples_step = b.step("test-examples", "Run example tests");
    examples_step.dependOn(&sum.step);
    examples_step.dependOn(&hello_tests.step);
}
