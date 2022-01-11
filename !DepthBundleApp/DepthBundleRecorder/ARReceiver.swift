/*
See LICENSE folder for this sample’s licensing information.

Abstract:
A utility class that receives processed depth information.
*/

import Foundation
import SwiftUI
import Combine
import ARKit

// Receive the newest AR data from an `ARReceiver`.
protocol ARDataReceiver: AnyObject {
    func onNewARData(arData: ARData)
}

//- Tag: ARData
// Store depth-related AR data.
final class ARData {
    var depthImage: CVPixelBuffer?
    var depthSmoothImage: CVPixelBuffer?
    var colorImage: CVPixelBuffer?
    var confidenceImage: CVPixelBuffer?
    var confidenceSmoothImage: CVPixelBuffer?
    var cameraIntrinsics = simd_float3x3()
    var cameraResolution = CGSize()
    // Additional vars
    var sampleTime : TimeInterval?
    var worldPose = simd_float4x4()
    var eulerAngles = simd_float3()
    var intrinsics  = simd_float3x3()
    var worldToCamera = simd_float4x4()
}

// Configure and run an AR session to provide the app with depth-related AR data.
final class ARReceiver: NSObject, ARSessionDelegate {
    var arData = ARData()
    var arSession = ARSession()
    weak var delegate: ARDataReceiver?
    
    // Configure and start the ARSession.
    override init() {
        super.init()
        arSession.delegate = self
        start()
    }
    
    // Configure the ARKit session.
    func start() {
        guard ARWorldTrackingConfiguration.supportsFrameSemantics([.sceneDepth]) else { return }
        // Enable both the `sceneDepth` and `smoothedSceneDepth` frame semantics.
        let config = ARWorldTrackingConfiguration()
        config.isAutoFocusEnabled = true
        config.frameSemantics = [.sceneDepth]
        arSession.run(config)
    }
    
    func pause() {
        arSession.pause()
    }
  
    // Send required data from `ARFrame` to the delegate class via the `onNewARData` callback.
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        if(frame.sceneDepth != nil) {
            arData.depthImage = frame.sceneDepth?.depthMap
            arData.colorImage = frame.capturedImage
            arData.confidenceImage = frame.sceneDepth?.confidenceMap
            arData.cameraResolution = frame.camera.imageResolution
            arData.sampleTime = frame.timestamp
            arData.worldPose = frame.camera.transform
            arData.eulerAngles = frame.camera.eulerAngles
            arData.worldToCamera = frame.camera.viewMatrix(for: UIInterfaceOrientation(rawValue: UIInterfaceOrientation.portrait.rawValue)!) // portrait upright
            arData.cameraIntrinsics = frame.camera.intrinsics
            delegate?.onNewARData(arData: arData)
        }
    }
}
