const std = @import("std");

pub fn build(b: *std.Build) void {
    const optimize = b.standardOptimizeOption(.{});
    const target = b.standardTargetOptions(.{});

    const proptest = b.addModule("proptest", .{
        .root_source_file = b.path("./proptest.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Examples
    const sum = addExample(b, b.path("examples/00_sum.zig"), optimize, target, proptest);
    const hello = addExample(b, b.path("examples/01_hello.zig"), optimize, target, proptest);
    const weird_distributive = addExample(b, b.path("examples/02_weird_distributive.zig"), optimize, target, proptest);

    const examples_step = b.step("test-examples", "Run example tests");
    examples_step.dependOn(&sum.step);
    examples_step.dependOn(&hello.step);
    examples_step.dependOn(&weird_distributive.step);
}

fn addExample(b: *std.Build, source: std.Build.LazyPath, optimize: std.builtin.OptimizeMode, target: std.Build.ResolvedTarget, proptest: *std.Build.Module) *std.Build.Step.Run {
    const example_module = b.createModule(.{
        .root_source_file = source,
        .optimize = optimize,
        .target = target,
        .imports = &.{
            .{ .name = "proptest", .module = proptest },
        },
    });

    const example_test_exe = b.addTest(.{
        .root_module = example_module,
    });

    return b.addRunArtifact(example_test_exe);
}
