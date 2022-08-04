const std = @import("std");
const proptest = @import("proptest");

const Integers = proptest.String(i32, .{
    .min_len = 2,
    .max_len = 2,
    .ranges = &.{
        .{ .min_max = .{ std.math.minInt(i32), std.math.maxInt(i32) } },
    },
});

test "integer sum is commutative" {
    try proptest.run(@src(), .{}, []const i32, Integers.strategy(), testSumIsCommutative);
}

fn testSumIsCommutative(integers: []const i32) !void {
    const x = integers[0] + integers[1];
    const y = integers[1] + integers[0];
    try std.testing.expectEqual(x, y);
}
