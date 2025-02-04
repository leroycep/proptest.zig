const std = @import("std");
const proptest = @import("proptest");

const Integer = proptest.Character(i32, &.{
    .{ .min_max = .{ 0, 8192 } },
    .{ .min_max = .{ -8192, 0 } },
});
const WeirdDistributiveFnInput = proptest.Tuple(&.{ i32, i32, i32 }, .{ Integer.strategy(), Integer.strategy(), Integer.strategy() });

test "weird distributive" {
    try proptest.run(testWeirdDistributive, WeirdDistributiveFnInput.strategy(), .{});
}

fn testWeirdDistributive(x: i32, y: i32, z: i32) !void {
    try std.testing.expectEqual(
        (x + y) * z,
        x * (y + z),
    );
}
