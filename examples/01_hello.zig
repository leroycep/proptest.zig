const std = @import("std");
const proptest = @import("proptest");

const Integers = proptest.String(i32, .{
    .ranges = &.{
        .{ .min_max = .{ std.math.minInt(i32), std.math.maxInt(i32) } },
    },
});

test "slice of integers is in ascending order after sorting" {
    try proptest.run(@src(), .{}, []const i32, Integers.strategy(), testIntegersAscending);
}

fn testIntegersAscending(integers: []const i32) !void {
    var sorted = try std.testing.allocator.dupe(i32, integers);
    defer std.testing.allocator.free(sorted);

    std.sort.sort(i32, sorted, {}, i32LessThan);

    if (sorted.len == 0) return;

    var prev_value = sorted[0];
    for (sorted[1..]) |value| {
        try std.testing.expect(prev_value <= value);
        prev_value = value;
    }
}

fn i32LessThan(_: void, a: i32, b: i32) bool {
    return a < b;
}
