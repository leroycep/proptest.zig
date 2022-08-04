
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.05";
    flake-utils.url = "github:numtide/flake-utils";
    zig-overlay.url = "github:arqv/zig-overlay";
    zig-overlay.inputs.nixpkgs.follows = "nixpkgs";
    nix-zig-builder.url = "github:leroycep/nix-zig-builder";
    nix-zig-builder.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    zig-overlay,
    nix-zig-builder,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {inherit system;};
      zig = zig-overlay.packages.${system}.master.latest;
      zig-builder = nix-zig-builder.packages.${system}.zig-builder;
    in rec {
      devShell = pkgs.mkShell {
        name = "my-finances-app-devshell";

        buildInputs = [ pkgs.theft ];
        nativeBuildInputs = [
          zig pkgs.entr pkgs.theft
        ];
      };

      formatter = pkgs.alejandra;
    });
}
