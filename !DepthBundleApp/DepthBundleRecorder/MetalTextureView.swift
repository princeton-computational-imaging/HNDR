/*
See LICENSE folder for this sample’s licensing information.

Abstract:
A view that displays a Metal-rendered depth visualization.
*/

import Foundation
import SwiftUI
import MetalKit
import Metal

// Display `MTLTextures` in an `MTKView` using SwiftUI.
//- Tag: MTKCoordinator`
class MTKCoordinator: NSObject, MTKViewDelegate {
    var content: MetalTextureContent
    let view: MTKView
    var pipelineState: MTLRenderPipelineState!
    var metalCommandQueue: MTLCommandQueue!
    
    init(content: MetalTextureContent, view: MTKView) {
        self.content = content
        self.view = view
        if let metalDevice = MTLCreateSystemDefaultDevice() {
            view.device = metalDevice
            self.metalCommandQueue = metalDevice.makeCommandQueue()!
        }
        super.init()
        
        prepareFunctions()
    }
    func prepareFunctions() {
        guard let metalDevice = view.device else { fatalError("Expected a Metal device.") }
        do {
            let library = metalDevice.makeDefaultLibrary()
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            pipelineDescriptor.vertexFunction = library!.makeFunction(name: "planeVertexShader")
            pipelineDescriptor.fragmentFunction = library!.makeFunction(name: "planeFragmentShader")
            pipelineDescriptor.vertexDescriptor = createPlaneMetalVertexDescriptor()
            pipelineState = try metalDevice.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            print("Unexpected error: \(error).")
        }
    }
    func createPlaneMetalVertexDescriptor() -> MTLVertexDescriptor {
        let mtlVertexDescriptor: MTLVertexDescriptor = MTLVertexDescriptor()
        // Store position in `attribute[[0]]`.
        mtlVertexDescriptor.attributes[0].format = .float2
        mtlVertexDescriptor.attributes[0].offset = 0
        mtlVertexDescriptor.attributes[0].bufferIndex = 0
        
        // Store texture coordinates in `attribute[[1]]`.
        mtlVertexDescriptor.attributes[1].format = .float2
        mtlVertexDescriptor.attributes[1].offset = 8
        mtlVertexDescriptor.attributes[1].bufferIndex = 0
        
        // Set stride to twice the `float2` bytes per vertex.
        mtlVertexDescriptor.layouts[0].stride = 2 * MemoryLayout<SIMD2<Float>>.stride
        mtlVertexDescriptor.layouts[0].stepRate = 1
        mtlVertexDescriptor.layouts[0].stepFunction = .perVertex
        
        return mtlVertexDescriptor
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        
    }
    
    // Draw a textured quad.
    func draw(in view: MTKView) {
        guard content.texture != nil else {
            print("There's no content to display.")
            return
        }
        guard let commandBuffer = metalCommandQueue.makeCommandBuffer() else { return }
        guard let passDescriptor = view.currentRenderPassDescriptor else { return }
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: passDescriptor) else { return }
        let vertexData: [Float] = [  -1, -1, 1, 1,
                                     1, -1, 1, 0,
                                     -1, 1, 0, 1,
                                     1, 1, 0, 0]
        encoder.setVertexBytes(vertexData, length: vertexData.count * MemoryLayout<Float>.stride, index: 0)
        encoder.setFragmentTexture(content.texture, index: 0)
        encoder.setRenderPipelineState(pipelineState)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        encoder.endEncoding()
        commandBuffer.present(view.currentDrawable!)
        commandBuffer.commit()
    }

}
//- Tag: MetalTextureView
struct MetalTextureView: UIViewRepresentable {
    var mtkView: MTKView
    var content: MetalTextureContent
    func makeCoordinator() -> MTKCoordinator {
        MTKCoordinator(content: content, view: mtkView)
    }
    func makeUIView(context: UIViewRepresentableContext<MetalTextureView>) -> MTKView {
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
    func updateUIView(_ uiView: MTKView, context: UIViewRepresentableContext<MetalTextureView>) {
        
    }
}
