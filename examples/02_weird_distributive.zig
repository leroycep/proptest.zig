const std = @import("std");
const proptest = @import("proptest");

const Integers = proptest.String(i32, .{
    .min_len = 3,
    .max_len = 3,
    .ranges = &.{
        .{ .min_max = .{ 0, 8192 } },
        .{ .min_max = .{ -8192, 0 } },
    },
});

test "weird distributive" {
    try proptest.run(testWeirdDistributive, Integers.strategy(), .{});
}

fn testWeirdDistributive(integers: []const i32) !void {
    const a = (integers[0] + integers[1]) * integers[2];
    const b = integers[0] * (integers[1] + integers[2]);
    try std.testing.expectEqual(a, b);
}
