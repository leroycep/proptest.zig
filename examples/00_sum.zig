const std = @import("std");
const proptest = @import("proptest");

const Integer = proptest.Character(i32, &.{
    .{ .min_max = .{ 0, std.math.maxInt(i32) } },
    .{ .min_max = .{ std.math.minInt(i32), 0 } },
});
const TwoInts = proptest.Tuple(&.{ i32, i32 }, .{ Integer.strategy(), Integer.strategy() });

test "integer addition is commutative" {
    try proptest.run(testAddIsCommutative, TwoInts.strategy(), .{});
}

fn testAddIsCommutative(x: i32, y: i32) !void {
    try std.testing.expectEqual(x + y, y + x);
}
