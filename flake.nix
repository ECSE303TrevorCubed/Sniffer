{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    treefmt-nix.url = "github:numtide/treefmt-nix";
    uv2nix.url = "github:pyproject-nix/uv2nix";
    pybuild.url = "github:pyproject-nix/build-system-pkgs";
    pyproject.url = "github:pyproject-nix/pyproject.nix";
    appimage.url = "github:ralismark/nix-appimage";
  };

  outputs =
    { self, ... }@inputs:
    inputs.flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import inputs.nixpkgs { inherit system; };
        treefmtconfig = inputs.treefmt-nix.lib.evalModule pkgs {
          projectRootFile = "flake.nix";
          programs = {
            alejandra.enable = true;
            toml-sort.enable = true;
            yamlfmt.enable = true;
            mdformat = {
              enable = true;
              plugins = ps: [
                ps.mdformat-gfm
              ];
              settings = {
                wrap = 88;
                end-of-line = "lf";
              };
            };
            clang-format.enable = true;
            shellcheck.enable = true;
            shfmt.enable = true;
            nixfmt.enable = true;
            typstyle.enable = true;
          };
          settings.formatter.shellcheck.excludes = [
            ".envrc"
          ];
        };
        python = pkgs.python314;
        workspace = inputs.uv2nix.lib.workspace.loadWorkspace {
          workspaceRoot = ./src;
        };
        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };
        pyprojectOverrides = final: prev: {
          bme68x = prev.bme68x.overrideAttrs (old: {
            postPatch = (old.postPatch or "") + ''
              sed -i "/^\s*sources =/s/, '[^']*\.h'//g" setup.py
            '';
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
              final.setuptools
            ];
            buildInputs = (old.buildInputs or [ ]) ++ [
            ];
          });
        };
        pythonBase = pkgs.callPackage inputs.pyproject.build.packages {
          inherit python;
        };
        pythonSet = pythonBase.overrideScope (
          pkgs.lib.composeManyExtensions [
            inputs.pybuild.overlays.wheel
            overlay
            pyprojectOverrides
          ]
        );
        venv = pythonSet.mkVirtualEnv "venv" workspace.deps.default;
        venvDev = pythonSet.mkVirtualEnv "venvDev" (workspace.deps.all or workspace.deps.default);
        inherit (pkgs.callPackages inputs.pyproject.build.util { }) mkApplication;
      in
      {
        formatter = treefmtconfig.config.build.wrapper;
        devShells.default = pkgs.mkShell {
          packages =
            with pkgs;
            [
              typst
              typstyle
              nil
              nixd
              uv
              swig
            ]
            ++ [
              wiringpi
              venvDev
            ]
            ++ pkgs.lib.optionals pkgs.stdenv.isLinux [ wiringpi ];

          env = {
            UV_NO_SYNC = "1";
            UV_PYTHON = pythonSet.python.interpreter;
            UV_PYTHON_DOWNLOADS = "never";
          };

          shellHook = ''
            PROJ_ROOT=$(git rev-parse --show-toplevel)/src
            export PYTHONPATH="$PROJ_ROOT:${venvDev}/lib/*/site-packages:$PYTHONPATH"
            export LD_LIBRARY_PATH="${pkgs.file}/lib:$LD_LIBRARY_PATH"
            ln -sfn ${venvDev} $PROJ_ROOT/.venv
          '';
        };
        packages =
          let
            sniff_apps =
              builtins.mapAttrs
                (
                  _: pkg:
                  pkg.overrideAttrs (oldAttrs: {
                    passthru = (oldAttrs.passthru or { }) // {
                      appimage = inputs.appimage.bundlers.${system}.default pkg;
                    };
                  })
                )
                {
                  sniff_py = pkgs.callPackage ./nix/build.nix { inherit mkApplication pythonSet venv; };
                };
            sniff_report = pkgs.callPackage ./nix/report.nix { };
            ci = pkgs.callPackage ./nix/ci.nix {
              apps = sniff_apps;
              reports = sniff_report;
            };
          in
          {
            default = sniff_apps.sniff_py;
            inherit (sniff_apps) sniff_py;
            inherit sniff_report ci;
          };
        checks = {
          formatting = treefmtconfig.config.build.check self;
          ruff-lint = pkgs.stdenvNoCC.mkDerivation {
            name = "ruff-lint";
            src = ./.;

            nativeBuildInputs = [ pkgs.ruff ];

            buildPhase = ''
              echo "Running Ruff linter checks..."
              ruff check ./src
            '';

            installPhase = "mkdir $out";
          };
          basedpyright-types = pkgs.stdenvNoCC.mkDerivation {
            name = "basedpyright-types";
            src = ./src;

            nativeBuildInputs = [
              venvDev
              pkgs.basedpyright
            ];

            buildPhase = ''
              echo "Running Basedpyright type checks..."
              ln -sfn ${venvDev} ./.venv
              # Exit non-zero on errors only: warnings are reportUnknown* noise
              # from untyped third-party libs (bme68x, onnxruntime, ...).
              basedpyright --level error
            '';

            installPhase = "mkdir $out";
          };
        };
      }
    );
}
