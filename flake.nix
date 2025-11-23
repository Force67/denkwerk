{
  description = "Lightweight primitives for building LLM powered Rust applications";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
          config.allowUnfree = true;
        };

        # Use latest stable Rust
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
        };

        # System-specific dependencies for GUI development
        guiDeps = with pkgs;
          (if stdenv.isLinux then [
            # Minimal dependencies - let the system handle the GUI libraries
            # since KDE/Wayland is already running
            pkg-config
            wayland-scanner

            # Font and text rendering (usually needed)
            fontconfig
            freetype

            # System libraries
            alsa-lib
            dbus

          ] else if stdenv.isDarwin then [
            # macOS dependencies
            darwin.apple_sdk.frameworks.Cocoa
            darwin.apple_sdk.frameworks.CoreFoundation
            darwin.apple_sdk.frameworks.CoreGraphics
            darwin.apple_sdk.frameworks.CoreVideo
            darwin.apple_sdk.frameworks.Metal
            darwin.apple_sdk.frameworks.QuartzCore
            libiconv
          ] else []);

        # Development dependencies
        devDeps = with pkgs; [
          rustToolchain
          cargo-watch
          cargo-edit
          cargo-flamegraph
          rust-analyzer-unwrapped

          # Build tools
          gcc
          gnumake
          cmake
          pkg-config

          # Code quality
          clippy
          rustfmt
        ];
      in
      {
        # Development shell with all dependencies
        devShells.default = pkgs.mkShell {
          buildInputs = guiDeps ++ devDeps;

          # Use system environment for GUI - don't override since KDE/Wayland is already working
          # Enable backtraces and debugging
          RUST_BACKTRACE = "1";
          RUST_LOG = "debug";

          # Minimal PKG_CONFIG_PATH for basic compilation
          PKG_CONFIG_PATH = with pkgs; lib.makeSearchPath "lib/pkgconfig" [
            fontconfig
            freetype
          ];

          shellHook = ''
            echo "🦀 Denkwerk Development Environment"
            echo "=================================="
            echo "Rust version: $(rustc --version)"
            echo ""
            echo "🖥️  Using system display server (KDE/Wayland detected)"
            echo "   The application should use your existing Wayland session"
            echo ""
            echo "Available binaries:"
            echo "  flow_editor   - GUI flow editor application"
            echo "  handoff-eval  - Handoff evaluation tool"
            echo ""
            echo "Useful commands:"
            echo "  cargo run --bin flow_editor"
            echo "  cargo build --release"
            echo "  cargo test"
            echo "  cargo clippy"
            echo ""
            echo "💡 If you encounter display issues, try:"
            echo "   export GDK_BACKEND=x11"
            echo "   export QT_QPA_PLATFORM=xcb"
            echo "   cargo run --bin flow_editor"
            echo ""
          '';
        };

        # Build the project
        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "denkwerk";
          version = "0.1.0";
          src = ./.;

          nativeBuildInputs = with pkgs; [
            pkg-config
            rustToolchain
          ] ++ (if stdenv.isLinux then [ wayland-scanner ] else []);

          buildInputs = guiDeps;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          meta = with pkgs.lib; {
            description = "Lightweight primitives for building LLM powered Rust applications";
            homepage = "https://github.com/yourusername/denkwerk";
            license = with licenses; [ mit asl20 ];
            maintainers = with maintainers; [ ];
            platforms = platforms.linux ++ platforms.darwin;
          };
        };

        # Flow editor application package
        packages.flow-editor = pkgs.rustPlatform.buildRustPackage {
          pname = "denkwerk-flow-editor";
          version = "0.1.0";
          src = ./.;

          nativeBuildInputs = with pkgs; [
            pkg-config
            rustToolchain
          ] ++ (if stdenv.isLinux then [ wayland-scanner ] else []);

          buildInputs = guiDeps;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          cargoBuildFlags = [ "--bin" "flow_editor" ];
          cargoInstallFlags = [ "--bin" "flow_editor" ];

          meta = with pkgs.lib; {
            description = "GUI flow editor for denkwerk";
            license = with licenses; [ mit asl20 ];
            platforms = platforms.linux ++ platforms.darwin;
          };
        };

        # Handoff evaluation tool package
        packages.handoff-eval = pkgs.rustPlatform.buildRustPackage {
          pname = "denkwerk-handoff-eval";
          version = "0.1.0";
          src = ./.;

          nativeBuildInputs = with pkgs; [
            pkg-config
            rustToolchain
          ] ++ (if stdenv.isLinux then [ wayland-scanner ] else []);

          buildInputs = guiDeps;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          cargoBuildFlags = [ "--bin" "handoff-eval" ];
          cargoInstallFlags = [ "--bin" "handoff-eval" ];

          meta = with pkgs.lib; {
            description = "Handoff evaluation tool for denkwerk";
            license = with licenses; [ mit asl20 ];
            platforms = platforms.linux ++ platforms.darwin;
          };
        };

        # Run the flow editor
        apps.flow-editor = flake-utils.lib.mkApp {
          drv = self.packages.${system}.flow-editor;
        };

        # Run the handoff evaluation tool
        apps.handoff-eval = flake-utils.lib.mkApp {
          drv = self.packages.${system}.handoff-eval;
        };

        # Default app is the flow editor
        apps.default = self.apps.${system}.flow-editor;
      });
}