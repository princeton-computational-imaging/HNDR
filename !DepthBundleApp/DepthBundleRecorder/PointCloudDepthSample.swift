/*
See LICENSE folder for this sample’s licensing information.

Abstract:
The single entry point for DepthBundleRecorder.
*/

import SwiftUI
@main
struct DepthBundleRecorder: App {
    var body: some Scene {
        WindowGroup {
            MetalDepthView()
        }
    }
}
