const std = @import("std");
const builtin = @import("builtin");

/// List of errors to control error flow
const ControlErrors = error{
    /// The current random values aren't working, try with a different set of random values.
    PropTestDiscard,
};

pub const RunOptions = struct {
    allocator: std.mem.Allocator = std.testing.allocator,
    cache_path: []const u8 = ".zig-cache/test-cases",
    max_iterations: usize = if (builtin.mode == .Debug) 1 else 100,
    print_value: bool = true,
    seed: ?usize = null,
};

const ShrinkControlErrors = error{
    ShrinkNoMoreTactics,
    ShrinkDeadEnd,
};

pub fn Strategy(comptime Input: type) type {
    return struct {
        create: fn (*Runner) anyerror!Input,
        shrink: fn (Input, *Runner, Runner.TacticIndex) anyerror!Input,
        destroy: fn (Input, *Runner) void,
        print: fn (Input) void,
    };
}

pub const Runner = struct {
    allocator: std.mem.Allocator,
    rand: std.Random,
    tactics: std.ArrayListUnmanaged(u32),

    pub const TacticIndex = enum(u32) { root, _ };

    pub fn tactic(this: *const @This(), tactic_index: TacticIndex) u32 {
        return this.tactics.items[@intFromEnum(tactic_index)];
    }

    pub fn tacticAfter(this: *@This(), idx: TacticIndex) !TacticIndex {
        const after = @intFromEnum(idx) + 1;
        if (after >= this.tactics.items.len) {
            try this.tactics.append(this.allocator, 0);
        }
        return @enumFromInt(after);
    }

    pub fn nextTactic(this: *@This(), tactic_index: TacticIndex) void {
        const idx: u32 = @intFromEnum(tactic_index);
        this.tactics.shrinkRetainingCapacity(idx + 1);
        this.tactics.items[idx] += 1;
    }

    pub fn incrementAnyTactic(this: *@This()) void {
        this.tactics.items[this.tactics.items.len - 1] += 1;
    }
};

pub fn run(src: std.builtin.SourceLocation, run_options: RunOptions, comptime Input: type, strategy: Strategy(Input), testFn: fn (Input) anyerror!void) !void {
    var cache = try std.fs.cwd().makeOpenPath(run_options.cache_path, .{});
    defer cache.close();

    const test_name = &cacheName(src);
    const seed = run_options.seed orelse try getInputU64(cache, test_name);

    // TODO: switch on return type
    var iterations: usize = 0;
    while (iterations < run_options.max_iterations) : (iterations += 1) {
        const iteration_seed = seed + iterations;
        var prng = std.Random.DefaultPrng.init(iteration_seed);
        var runner = Runner{ .allocator = run_options.allocator, .rand = prng.random(), .tactics = .{} };
        defer runner.tactics.deinit(runner.allocator);

        const input = try strategy.create(&runner);

        std.debug.print("\r{s} iter {}/{}", .{ src.fn_name, iterations, run_options.max_iterations });
        testFn(input) catch |initial_error| switch (initial_error) {
            error.PropTestDiscard => continue,
            else => {
                // Try to shrink test case
                runner.tactics.shrinkRetainingCapacity(0);
                try runner.tactics.append(runner.allocator, 0);
                var current_input = input;
                var current_error = initial_error;

                shrinking: while (true) {
                    const new_input = strategy.shrink(current_input, &runner, .root) catch |err| switch (err) {
                        error.ShrinkDeadEnd => {
                            runner.nextTactic(.root);
                            continue :shrinking;
                        },
                        error.ShrinkNoMoreTactics => break :shrinking,
                        else => |e| return e,
                    };

                    testFn(new_input) catch |e| if (e == initial_error) {
                        std.debug.print("{s}:{} same as initial error: {}\n", .{ @src().file, @src().line, e });
                        // We got the same error back out! Continue simplifying with the new input, resetting the tactics we're using
                        runner.tactics.shrinkRetainingCapacity(0);
                        runner.tactics.appendAssumeCapacity(0);
                        strategy.destroy(current_input, &runner);
                        current_input = new_input;
                        current_error = e;

                        continue;
                    } else {
                        // We got a different error: treat it the same as succeeding (error didn't reproduce)
                        //std.debug.print("{s}:{} different error: {}\n", .{ @src().file, @src().line, e });
                    };
                    //std.debug.print("{s}:{} trying different tactic: \n", .{ @src().file, @src().line });

                    // The test didn't fail, so our simplification didn't work.
                    // Change tactics and continue.
                    runner.incrementAnyTactic();
                    strategy.destroy(new_input, &runner);
                }

                // Print input
                std.debug.print("{s} failed with error: {}\n", .{ src.fn_name, current_error });
                if (run_options.print_value) {
                    strategy.print(current_input);
                    std.debug.print("\n", .{});
                }

                std.debug.print("Seed for value {}\n", .{iteration_seed});

                strategy.destroy(current_input, &runner);
                return current_error;
            },
        };

        strategy.destroy(input, &runner);
    }

    if (run_options.seed == null) {
        cleanTestCache(cache, test_name);
    }
}

/// Get an u64 and store it in the test cases cache. This allows us to redo the same test case if it fails.
pub fn getInputU64(cache: std.fs.Dir, test_name: []const u8) !u64 {
    const test_case = cache.readFileAlloc(std.testing.allocator, test_name, @sizeOf(u64)) catch |e| switch (e) {
        error.FileTooBig => {
            std.debug.print("Test case in cache too large\n", .{});
            return error.UnexpectedValueInCache;
        },

        // Generate a test case
        error.FileNotFound => {
            const new_test_case = std.crypto.random.int(u64);
            try cache.writeFile(.{
                .sub_path = test_name,
                .data = std.mem.asBytes(&new_test_case),
            });
            return new_test_case;
        },

        else => return e,
    };
    defer std.testing.allocator.free(test_case);
    if (test_case.len < @sizeOf(u64)) {
        std.debug.print("Test case in cache too small\n", .{});
        return error.UnexpectedValueInCache;
    }

    return @bitCast(test_case[0..@sizeOf(u64)].*);
}

/// The test succeeded, we remove the cached input and do a different test case
pub fn cleanTestCache(cache: std.fs.Dir, test_name: []const u8) void {
    cache.deleteFile(test_name) catch return;
}

pub const CACHE_NAME_LEN = std.base64.url_safe_no_pad.Encoder.calcSize(16);

pub fn cacheName(src: std.builtin.SourceLocation) [CACHE_NAME_LEN]u8 {
    var hash: [16]u8 = undefined;
    std.crypto.hash.Blake3.hash(src.fn_name, &hash, .{});

    var name: [std.base64.url_safe_no_pad.Encoder.calcSize(16)]u8 = undefined;
    _ = std.base64.url_safe_no_pad.Encoder.encode(&name, &hash);

    return name;
}

// A random interface that logs the values it returns for use later
const RandomRecorder = struct {
    allocator: std.mem.Allocator,
    parent: std.rand.Random,
    record: std.SegmentedList(u8, 0) = .{},
    num_bytes_not_recorded: usize = 0,

    pub fn deinit(this: *@This()) void {
        this.record.deinit(this.allocator);
        if (this.num_bytes_not_recorded > 0) {
            std.debug.print("RandomRecorder missed recording {} bytes", .{this.num_bytes_not_recorded});
        }
    }

    pub fn random(this: *@This()) std.rand.Random {
        return std.rand.Random.init(this, fill);
    }

    pub fn fill(this: *@This(), buf: []u8) void {
        this.parent.bytes(buf);
        this.record.appendSlice(this.allocator, buf) catch {
            this.num_bytes_not_recorded += buf.len;
        };
    }
};

// A random interface that logs the values it returns for use later
const RandomReplayer = struct {
    record: *const std.SegmentedList(u8, 0),
    index: usize = 0,

    /// The maximum index to return recorded data from. Panics if more data is requested.
    max_index: usize,

    /// The index where we want to start simplifying the data. Replayer will return 0 after this.
    begin_simplify_at_index: ?usize = null,

    pub fn init(recorder: *const RandomRecorder) @This() {
        return @This(){
            .record = &recorder.record,
            .max_index = recorder.record.count(),
        };
    }

    pub fn random(this: *@This()) std.rand.Random {
        return std.rand.Random.init(this, fill);
    }

    pub fn fill(this: *@This(), buf: []u8) void {
        for (buf) |*c| {
            if (this.index >= this.max_index) {
                std.debug.panic("Random data requested beyond recorded data", .{});
            }
            if (this.index >= this.begin_simplify_at_index) {
                c.* = 0;
            } else {
                c.* = this.record.at(this.index);
            }
            this.index += 1;
        }
    }
};

pub fn String(comptime T: type, comptime options: struct {
    ranges: []const Range(T) = &.{.{ .min_max = .{ 0, std.math.maxInt(T) } }},
    min_len: usize = 0,
    max_len: usize = 50 * 1024,
}) type {
    const StringCharacter = Character(T, options.ranges);

    return struct {
        pub fn strategy() Strategy([]const T) {
            return .{
                .create = create,
                .destroy = destroy,
                .shrink = shrink,
                .print = print,
            };
        }

        pub fn create(runner: *Runner) ![]const T {
            const len = if (options.min_len == options.max_len) options.min_len else runner.rand.intRangeLessThan(usize, options.min_len, options.max_len);
            const buf = try runner.allocator.alloc(T, len);
            for (buf) |*element| {
                element.* = try StringCharacter.create(runner);
            }
            return buf;
        }

        pub fn destroy(buf: []const T, runner: *Runner) void {
            runner.allocator.free(buf);
        }

        const Tactic = enum(u32) {
            take_front_half,
            take_back_half,
            simplify,
            simplify_front_half,
            simplify_back_half,
            remove_last_char,
            remove_first_char,
            simplify_last_char,
            simplify_first_char,
            _,
        };

        pub fn shrink(buf: []const T, runner: *Runner, tactic_index: Runner.TacticIndex) anyerror![]const T {
            if (buf.len < options.min_len) return error.ShrinkNoMoreTactics;
            const tactic: Tactic = @enumFromInt(runner.tactic(tactic_index));
            switch (tactic) {
                .take_front_half,
                .take_back_half,
                => {
                    if (buf.len / 2 < options.min_len) return error.ShrinkDeadEnd;
                    const buf_to_copy = switch (tactic) {
                        .take_front_half => buf[0 .. buf.len / 2],
                        .take_back_half => buf[buf.len / 2 ..],
                        else => unreachable,
                    };
                    if (buf_to_copy.len == buf.len) return error.ShrinkDeadEnd;
                    return try runner.allocator.dupe(T, buf_to_copy);
                },
                .simplify,
                .simplify_front_half,
                .simplify_back_half,
                => {
                    const new = try runner.allocator.dupe(T, buf);
                    var should_free = true;
                    defer if (should_free) runner.allocator.free(new);

                    const to_simplify = switch (tactic) {
                        .simplify => new,
                        .simplify_front_half => new[buf.len / 2 ..],
                        .simplify_back_half => new[0 .. buf.len / 2],
                        else => unreachable,
                    };
                    if (to_simplify.len == 0) return error.ShrinkDeadEnd;

                    const char_tactic = try runner.tacticAfter(tactic_index);

                    while (true) {
                        var any_shrunk = false;
                        for (to_simplify) |*element| {
                            element.* = StringCharacter.shrink(element.*, runner, char_tactic) catch |err| switch (err) {
                                error.ShrinkDeadEnd => continue,
                                error.ShrinkNoMoreTactics => return error.ShrinkDeadEnd,
                                else => return err,
                            };
                            any_shrunk = true;
                        }
                        if (!any_shrunk) {
                            runner.nextTactic(char_tactic);
                            @memcpy(new, buf);
                            continue;
                        }
                        break;
                    }

                    should_free = false;
                    return new;
                },
                .remove_last_char => {
                    if (buf.len - 1 < options.min_len) return error.ShrinkDeadEnd;
                    return try runner.allocator.dupe(T, buf[0 .. buf.len - 1]);
                },
                .remove_first_char => {
                    if (buf.len - 1 < options.min_len) return error.ShrinkDeadEnd;
                    return try runner.allocator.dupe(T, buf[1..]);
                },

                .simplify_last_char,
                .simplify_first_char,
                => {
                    const to_simplify = switch (tactic) {
                        .simplify_last_char => buf.len - 1,
                        .simplify_first_char => 0,
                        else => unreachable,
                    };
                    const char_tactic = try runner.tacticAfter(tactic_index);
                    const new_char = while (true) {
                        break StringCharacter.shrink(buf[to_simplify], runner, char_tactic) catch |err| switch (err) {
                            error.ShrinkDeadEnd => {
                                runner.nextTactic(char_tactic);
                                continue;
                            },
                            error.ShrinkNoMoreTactics => return error.ShrinkDeadEnd,
                            else => |e| return e,
                        };
                    } else unreachable;

                    const new = try runner.allocator.dupe(T, buf);
                    new[to_simplify] = new_char;

                    return new;
                },

                _ => return error.ShrinkNoMoreTactics,
            }
        }

        pub fn print(string: []const T) void {
            std.debug.print("{any}\n", .{string});
        }
    };
}

pub fn Range(comptime T: type) type {
    return union(enum) {
        list: []const T,
        min_max: [2]T,

        pub fn valueAt(this: @This(), index: usize) T {
            switch (this) {
                .list => |l| return l[index],
                .min_max => |r| return r[0] + @as(T, @intCast(index)),
            }
        }

        pub fn size(this: @This()) std.meta.Int(.unsigned, @typeInfo(T).int.bits) {
            switch (this) {
                .list => |l| return @intCast(l.len),
                .min_max => |r| {
                    std.debug.assert(r[0] < r[1]);
                    const WideInt = std.meta.Int(@typeInfo(T).int.signedness, @typeInfo(T).int.bits + 1);
                    const left: WideInt = r[0];
                    const right: WideInt = r[1];
                    return @intCast(right - left);
                },
            }
        }

        pub fn indexOf(this: @This(), t: T) ?usize {
            switch (this) {
                .list => |l| return std.mem.indexOfScalar(T, l, t),
                .min_max => |r| {
                    std.debug.assert(r[0] < r[1]);
                    if (r[0] <= t and t < r[1]) {
                        return @intCast(t - r[0]);
                    }
                    return null;
                },
            }
        }
    };
}

pub fn Character(comptime T: type, comptime ranges: []const Range(T)) type {
    std.debug.assert(ranges.len > 0);
    // TODO: Ensure that none of the ranges overlap
    var total: usize = 0;
    for (ranges) |range| {
        total += range.size();
    }
    const total_number_of_characters = total;
    return struct {
        fn create(runner: *Runner) !T {
            if (runner.rand.int(u4) == 0) {
                if (runner.rand.boolean()) {
                    return std.math.maxInt(T);
                } else {
                    return std.math.minInt(T);
                }
            }
            if (ranges.len == 1 and ranges[0] == .min_max) {
                return runner.rand.intRangeAtMost(T, ranges[0].min_max[0], ranges[0].min_max[1]);
            }
            const index = runner.rand.uintLessThan(usize, total_number_of_characters);
            return indexToCharacter(index);
        }

        fn indexToCharacter(index: usize) T {
            var range_start: usize = 0;
            for (ranges) |range| {
                const range_end = range_start + range.size();
                if (range_start <= index and index < range_end) {
                    return range.valueAt(index - range_start);
                }
                range_start = range_end;
            } else {
                std.debug.panic("index not in ranges! {}", .{index});
            }
        }

        fn characterToIndex(character: T) ?usize {
            var range_start: usize = 0;
            for (ranges) |range| {
                if (range.indexOf(character)) |idx_in_range| {
                    return range_start + idx_in_range;
                }
                range_start += range.size();
            }
            return null;
        }

        fn destroy(_: T, _: std.mem.Allocator) void {}

        const Tactic = enum(u32) {
            change_to_first,
            set_to_half,
            decrement,
            _,
        };

        fn shrink(current: T, runner: *Runner, tactic_index: Runner.TacticIndex) anyerror!T {
            const tactic: Tactic = @enumFromInt(runner.tactic(tactic_index));
            switch (tactic) {
                .change_to_first => {
                    const first = ranges[0].valueAt(0);
                    if (current == first) return error.ShrinkDeadEnd;
                    return first;
                },
                .set_to_half => {
                    const current_idx = characterToIndex(current) orelse {
                        std.debug.panic("Generated character not in ranges! (0x{x}, {})", .{ current, current });
                    };
                    if (current_idx == 0) return error.ShrinkDeadEnd;
                    return indexToCharacter(current_idx / 2);
                },
                .decrement => {
                    const current_idx = characterToIndex(current) orelse {
                        std.debug.panic("Generated character not in ranges! (0x{x}, {})", .{ current, current });
                    };
                    if (current_idx == 0) return error.ShrinkDeadEnd;
                    return indexToCharacter(current_idx - 1);
                },
                _ => return error.ShrinkNoMoreTactics,
            }
        }

        fn print(value: T) void {
            std.debug.print("\'{'}\' ({}, 0x{x})\n", .{ std.zig.fmtEscapes(value), value, value });
        }
    };
}
