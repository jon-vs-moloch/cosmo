# Cosmo macOS app

Cosmo is intentionally a terminal-first experimental harness. `Cosmo.app` is
a Finder-launchable native macOS wrapper that opens the bundled CLI in Terminal
and writes the same stdout/stderr stream to `~/Library/Logs/Cosmo/cosmo.log`.
It is not a separate GUI implementation.

The bundled package includes the product modules for the local atomic control
plane: context, planning, sandbox, worker, integration, control, chat, tools,
and the atomic harness — not only the older reasoning-mode sidecar.

## Build

From a clean checkout on macOS:

```sh
# Remove only the known local target if you need a rebuild of the default path
rm -rf dist/Cosmo.app
scripts/package_macos_app.sh
open dist/Cosmo.app
```

The build creates `dist/Cosmo.app`; it refuses to overwrite an existing target.
Choose another path with `--output`, select the build Python with `--python`,
or validate prerequisites without writing with `--dry-run`. The icon set is
resized from the committed `resources/cosmo-icon.png` by
`generate_macos_icon.sh`; `resources/cosmo-icon.svg` is its editable source
(re-render the PNG with `sips` after editing it). The bundle also carries
`LICENSE` and `NOTICE` under `Contents/Resources/`.

## Runtime boundary

The app bundles Cosmo source and benchmark assets, plus its native launcher.
It deliberately does **not** bundle a Python framework: at launch it requires
a user-installed Python 3.11+ (`python3`, or `COSMO_PYTHON`). This makes the
bundle relocatable while failing closed and visibly if its small host-runtime
requirement is not met.

No credential, token, home-directory auth file, or environment file is copied
into the bundle. The app does not read or create provider login state. At
runtime it connects only to the operator-owned loopback LM Studio server; the
server and its model lifecycle remain outside the app bundle.

The unsigned bundle is for local development/testing only. Packaging does not
sign, notarize, publish, or install it.

## Validate

```sh
scripts/package_macos_app.sh --dry-run
scripts/check_macos_app.sh dist/Cosmo.app
plutil -lint dist/Cosmo.app/Contents/Info.plist
codesign --display --verbose=1 dist/Cosmo.app  # reports unsigned for dev builds
```

Opening the app with no command shows the bundled command guide in Terminal.
The first useful check is a local LM Studio capability probe:

```sh
dist/Cosmo.app/Contents/Resources/launch_cosmo.command \
  doctor --model <loaded-instance> --probe
```
