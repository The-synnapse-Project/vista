{
  description = "Build a cargo project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

    crane.url = "github:ipetkov/crane";

    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.rust-analyzer-src.follows = "";
    };

    flake-utils.url = "github:numtide/flake-utils";

    advisory-db = {
      url = "github:rustsec/advisory-db";
      flake = false;
    };
  };

  outputs = {
    self,
    nixpkgs,
    crane,
    fenix,
    flake-utils,
    advisory-db,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = nixpkgs.legacyPackages.${system};

      inherit (pkgs) lib;

      craneLib = crane.mkLib pkgs;
      src = craneLib.cleanCargoSource ./.;

      # Common arguments can be set here to avoid repeating them later
      commonArgs = {
        inherit src;
        strictDeps = true;

        nativeBuildInputs = with pkgs; [
          pkg-config
          clang
        ];

        buildInputs = with pkgs;
          [
            alsa-lib
            speechd
            openssl
            glib
            ffmpeg
            libva
            ((opencv.overrideAttrs (old: {
                cmakeFlags = old.cmakeFlags ++ [(lib.cmakeFeature "CMAKE_CXX_STANDARD" "17")];
              })).override {
                enableGtk3 = true;
                enableGStreamer = true;
                enableGtk2 = true;
                enableContrib = true;
              })
            pkg-config
            clang
            protobuf_21
            protobufc
          ]
          ++ lib.optionals pkgs.stdenv.isDarwin [
            # Additional darwin specific inputs can be set here
            pkgs.libiconv
          ];

        # Additional environment variables can be set directly
        # MY_CUSTOM_VAR = "some value";
        LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
        PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";
      };

      craneLibLLvmTools =
        craneLib.overrideToolchain
        (fenix.packages.${system}.complete.withComponents [
          "cargo"
          "llvm-tools"
          "rustc"
        ]);

      # Build *just* the cargo dependencies, so we can reuse
      # all of that work (e.g. via cachix) when running in CI
      cargoArtifacts = craneLib.buildDepsOnly commonArgs;

      # Build the actual crate itself, reusing the dependency
      # artifacts from above.
      my-crate = craneLib.buildPackage (commonArgs
        // {
          inherit cargoArtifacts;
        });
    in {
      checks = {
        # Build the crate as part of `nix flake check` for convenience
        inherit my-crate;

        # Run clippy (and deny all warnings) on the crate source,
        # again, reusing the dependency artifacts from above.
        #
        # Note that this is done as a separate derivation so that
        # we can block the CI if there are issues here, but not
        # prevent downstream consumers from building our crate by itself.
        my-crate-clippy = craneLib.cargoClippy (commonArgs
          // {
            inherit cargoArtifacts;
            cargoClippyExtraArgs = "--all-targets -- --deny warnings";
          });

        my-crate-doc = craneLib.cargoDoc (commonArgs
          // {
            inherit cargoArtifacts;
          });

        # Check formatting
        my-crate-fmt = craneLib.cargoFmt {
          inherit src;
        };

        my-crate-toml-fmt = craneLib.taploFmt {
          src = pkgs.lib.sources.sourceFilesBySuffices src [".toml"];
          # taplo arguments can be further customized below as needed
          # taploExtraArgs = "--config ./taplo.toml";
        };

        # Audit dependencies
        my-crate-audit = craneLib.cargoAudit {
          inherit src advisory-db;
        };

        # Audit licenses
        # my-crate-deny = craneLib.cargoDeny {
        #   inherit src;
        # };

        # Run tests with cargo-nextest
        # Consider setting `doCheck = false` on `my-crate` if you do not want
        # the tests to run twice
        my-crate-nextest = craneLib.cargoNextest (commonArgs
          // {
            inherit cargoArtifacts;
            partitions = 1;
            partitionType = "count";
            cargoNextestPartitionsExtraArgs = "--no-tests=pass";
          });
      };

      packages =
        {
          default = my-crate;
        }
        // lib.optionalAttrs (!pkgs.stdenv.isDarwin) {
          my-crate-llvm-coverage = craneLibLLvmTools.cargoLlvmCov (commonArgs
            // {
              inherit cargoArtifacts;
            });
        };

      apps.default = flake-utils.lib.mkApp {
        drv = my-crate;
      };

      devShells.default = craneLib.devShell {
        # Inherit inputs from checks.
        checks = self.checks.${system};

        # Additional dev-shell environment variables can be set directly
        # MY_CUSTOM_DEVELOPMENT_VAR = "something else";
        LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
        PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";

        # Extra inputs can be added here; cargo and rustc are provided by default.
        packages = with pkgs; [
          # pkgs.ripgrep
          gdb
          alsa-lib
          speechd
          openssl
          glib
          ffmpeg
          libva
          ((opencv.overrideAttrs (old: {
              cmakeFlags = old.cmakeFlags ++ [(lib.cmakeFeature "CMAKE_CXX_STANDARD" "17")];
            })).override {
              enableGtk3 = true;
              enableGStreamer = true;
              enableGtk2 = true;
              enableContrib = true;
            })
          pkg-config
          clang
          protobuf_21
          protobufc
        ];
      };
    });
}
