/*
See LICENSE folder for this sample’s licensing information.

Abstract:
A view that displays scene depth information.
*/

import Foundation
import SwiftUI
import MetalKit
import Metal

//- Tag: CoordinatorDepth
final class CoordinatorDepth: MTKCoordinator {
    @Binding var confSelection: Int
    init(mtkView: MTKView, depthContent: MetalTextureContent, confSelection: Binding<Int>) {
        self._confSelection = confSelection
        super.init(content: depthContent, view: mtkView)
    }
    override func prepareFunctions() {
        guard let metalDevice = view.device else { fatalError("Expected a Metal device.") }
        do {
            let library = metalDevice.makeDefaultLibrary()
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            pipelineDescriptor.vertexFunction = library!.makeFunction(name: "planeVertexShader")
            pipelineDescriptor.fragmentFunction = library!.makeFunction(name: "planeFragmentShaderDepth")
            pipelineDescriptor.vertexDescriptor = createPlaneMetalVertexDescriptor()
            pipelineState = try metalDevice.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            print("Unexpected error: \(error).")
        }
    }

}

struct MetalTextureViewDepth: UIViewRepresentable {
    var mtkView: MTKView
    var content: MetalTextureContent
    
    @Binding var confSelection: Int
    func makeCoordinator() -> CoordinatorDepth {
        CoordinatorDepth(mtkView: mtkView, depthContent: content, confSelection: $confSelection)
    }
    
    func makeUIView(context: UIViewRepresentableContext<MetalTextureViewDepth>) -> MTKView {
        mtkView.delegate = context.coordinator
        mtkView.preferredFramesPerSecond = 120
        mtkView.backgroundColor = context.environment.colorScheme == .dark ? .black : .white
        mtkView.isOpaque = true
        mtkView.framebufferOnly = false
        mtkView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        mtkView.drawableSize = mtkView.frame.size
        mtkView.enableSetNeedsDisplay = false
        mtkView.colorPixelFormat = .bgra8Unorm
        return mtkView
    }
    
    // `UIViewRepresentable` requires this implementation; however, the sample
    // app doesn't use it. Instead, `MTKView.delegate` handles display updates.
    func updateUIView(_ uiView: MTKView, context: UIViewRepresentableContext<MetalTextureViewDepth>) {
        
    }
}
