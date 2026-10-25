import AppKit
import Foundation

// Cosmo deliberately remains a terminal-first experimental workbench.  The
// native app is only a Finder-friendly launcher; it never handles credentials
// or embeds an inference provider SDK.
let bundle = Bundle.main
guard let command = bundle.url(forResource: "launch_cosmo", withExtension: "command") else {
    let alert = NSAlert()
    alert.messageText = "Cosmo is incomplete"
    alert.informativeText = "The bundled Cosmo launcher script is missing. Rebuild the app from a clean checkout."
    alert.runModal()
    exit(1)
}

let opener = Process()
opener.executableURL = URL(fileURLWithPath: "/usr/bin/open")
opener.arguments = ["-a", "Terminal", command.path]

do {
    try opener.run()
} catch {
    let alert = NSAlert()
    alert.messageText = "Cosmo could not open Terminal"
    alert.informativeText = "\(error.localizedDescription)"
    alert.runModal()
    exit(1)
}
