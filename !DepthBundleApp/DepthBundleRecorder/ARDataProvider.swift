/*
See LICENSE folder for this sample’s licensing information.

Abstract:
A utility class that provides processed depth information.
*/

import Foundation
import SwiftUI
import Combine
import ARKit
import Accelerate
import MetalPerformanceShaders

// Wrap the `MTLTexture` protocol to reference outputs from ARKit.
final class MetalTextureContent {
    var texture: MTLTexture?
}

// Enable `CVPixelBuffer` to output an `MTLTexture`.
extension CVPixelBuffer {
    
    func texture(withFormat pixelFormat: MTLPixelFormat, planeIndex: Int, addToCache cache: CVMetalTextureCache) -> MTLTexture? {
        
        let width = CVPixelBufferGetWidthOfPlane(self, planeIndex)
        let height = CVPixelBufferGetHeightOfPlane(self, planeIndex)
        
        var cvtexture: CVMetalTexture?
        _ = CVMetalTextureCacheCreateTextureFromImage(nil, cache, self, nil, pixelFormat, width, height, planeIndex, &cvtexture)
        let texture = CVMetalTextureGetTexture(cvtexture!)
        
        return texture
        
    }
    
}

// Collect AR data using a lower-level receiver. This class converts AR data
// to a Metal texture, optionally upscaling depth data using a guided filter,
// and implements `ARDataReceiver` to respond to `onNewARData` events.
final class ARProvider: ARDataReceiver, ObservableObject {
    // Set the destination resolution for the upscaled algorithm.
    let upscaledWidth = 256
    let upscaledHeight = 192

    // Set the original depth size.
    let origDepthWidth = 256
    let origDepthHeight = 192

    // Set the original color size.
    let origColorWidth = 1920
    let origColorHeight = 1440
    
    // Set the guided filter constants.
    let guidedFilterEpsilon: Float = 0.004
    let guidedFilterKernelDiameter = 5
    
    // For recording LiDAR bundle
    var recordScene = true
    var bundleSize : Int = 15
    var frameCount = 99999
    @Published var bundleFolder : URL?
    
    let arReceiver = ARReceiver()
    @Published var lastArData: ARData?
    let depthContent = MetalTextureContent()
    let confidenceContent = MetalTextureContent()
    let colorYContent = MetalTextureContent()
    let colorCbCrContent = MetalTextureContent()
    let upscaledCoef = MetalTextureContent()
    let colorRGBContent = MetalTextureContent()
    let upscaledConfidence = MetalTextureContent()
    
    let coefTexture: MTLTexture
    let destDepthTexture: MTLTexture
    let destConfTexture: MTLTexture
    let colorRGBTexture: MTLTexture
    let colorRGBTextureDownscaled: MTLTexture
    let colorRGBTextureDownscaledLowRes: MTLTexture
    
    // Enable or disable depth upsampling.
    public var isToUpsampleDepth: Bool = false {
        didSet {
            processLastArData()
        }
    }
    
    // Enable or disable smoothed-depth upsampling.
    public var isUseSmoothedDepthForUpsampling: Bool = false {
        didSet {
            processLastArData()
        }
    }
    var textureCache: CVMetalTextureCache?
    let metalDevice: MTLDevice?
    let guidedFilter: MPSImageGuidedFilter?
    let mpsScaleFilter: MPSImageBilinearScale?
    let commandQueue: MTLCommandQueue?
    let pipelineStateCompute: MTLComputePipelineState?
    
    // Create an empty texture.
    static func createTexture(metalDevice: MTLDevice, width: Int, height: Int, usage: MTLTextureUsage, pixelFormat: MTLPixelFormat) -> MTLTexture {
        let descriptor: MTLTextureDescriptor = MTLTextureDescriptor()
        descriptor.pixelFormat = pixelFormat
        descriptor.width = width
        descriptor.height = height
        descriptor.usage = usage
        let resTexture = metalDevice.makeTexture(descriptor: descriptor)
        return resTexture!
    }
    
    // Start or resume the stream from ARKit.
    func start() {
        arReceiver.start()
    }
    
    // Pause the stream from ARKit.
    func pause() {
        arReceiver.pause()
    }
    
    // Initialize the MPS filters, metal pipeline, and Metal textures.
    init?() {
        do {
            metalDevice = MTLCreateSystemDefaultDevice()
            CVMetalTextureCacheCreate(nil, nil, metalDevice!, nil, &textureCache)
            guidedFilter = MPSImageGuidedFilter(device: metalDevice!, kernelDiameter: guidedFilterKernelDiameter)
            guidedFilter?.epsilon = guidedFilterEpsilon
            mpsScaleFilter = MPSImageBilinearScale(device: metalDevice!)
            commandQueue = metalDevice!.makeCommandQueue()
            let lib = metalDevice!.makeDefaultLibrary()
            let convertYUV2RGBFunc = lib!.makeFunction(name: "convertYCbCrToRGBA")
            pipelineStateCompute = try metalDevice!.makeComputePipelineState(function: convertYUV2RGBFunc!)
            // Initialize the working textures.
            coefTexture = ARProvider.createTexture(metalDevice: metalDevice!, width: origDepthWidth, height: origDepthHeight,
                                                   usage: [.shaderRead, .shaderWrite], pixelFormat: .rgba32Float)
            destDepthTexture = ARProvider.createTexture(metalDevice: metalDevice!, width: upscaledWidth, height: upscaledHeight,
                                                        usage: [.shaderRead, .shaderWrite], pixelFormat: .r32Float)
            destConfTexture = ARProvider.createTexture(metalDevice: metalDevice!, width: upscaledWidth, height: upscaledHeight,
                                                       usage: [.shaderRead, .shaderWrite], pixelFormat: .r8Unorm)
            colorRGBTexture = ARProvider.createTexture(metalDevice: metalDevice!, width: origColorWidth, height: origColorHeight,
                                                       usage: [.shaderRead, .shaderWrite], pixelFormat: .rgba32Float)
            colorRGBTextureDownscaled = ARProvider.createTexture(metalDevice: metalDevice!, width: upscaledWidth, height: upscaledHeight,
                                                                 usage: [.shaderRead, .shaderWrite], pixelFormat: .rgba32Float)
            colorRGBTextureDownscaledLowRes = ARProvider.createTexture(metalDevice: metalDevice!, width: origDepthWidth, height: origDepthHeight,
                                                                       usage: [.shaderRead, .shaderWrite], pixelFormat: .rgba32Float)
            upscaledCoef.texture = coefTexture
            upscaledConfidence.texture = destConfTexture
            colorRGBContent.texture = colorRGBTextureDownscaled
            
            // Set the delegate for ARKit callbacks.
            arReceiver.delegate = self
            
        } catch {
            print("Unexpected error: \(error).")
            return nil
        }
    }
    
    func writeImageYUV(pixelBuffer: CVPixelBuffer, fileNameSuffix : String) {
        // Image is 2 Plane YUV, shape HxW, H/2 x W/2

        guard CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly) == noErr else { return }
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        guard let srcPtrP0 = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0) else {
          return
        }
        guard let srcPtrP1 = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1) else {
          return
        }

        let rowBytesP0 : Int = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
        let rowBytesP1 : Int = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1)
        let widthP0 = Int(CVPixelBufferGetWidthOfPlane(pixelBuffer, 0))
        let widthP1 = Int(CVPixelBufferGetWidthOfPlane(pixelBuffer, 1))
        let heightP0 = Int(CVPixelBufferGetHeightOfPlane(pixelBuffer, 0))
        let heightP1 = Int(CVPixelBufferGetHeightOfPlane(pixelBuffer, 1))

        let uint8PointerP0 = srcPtrP0.bindMemory(to: UInt8.self, capacity: heightP0 * rowBytesP0)
        let uint8PointerP1 = srcPtrP1.bindMemory(to: UInt8.self, capacity: heightP1 * rowBytesP1)

        let s = "image_P0_\(widthP0)x\(heightP0)_P1_\(widthP1)x\(heightP1)_\(fileNameSuffix)"
        let fileURL = URL(fileURLWithPath: s, relativeTo: bundleFolder).appendingPathExtension("bin")

        let stream = OutputStream(url: fileURL, append: false)
        stream?.open()

        for y in 0 ..< heightP0{
            stream?.write(uint8PointerP0 + (y * rowBytesP0), maxLength: Int(rowBytesP0))
        }
        
        for y in 0 ..< heightP1{
            stream?.write(uint8PointerP1 + (y * rowBytesP1), maxLength: Int(rowBytesP1))
        }

        stream?.close()
    }
    
    func writeDepth(pixelBuffer: CVPixelBuffer, fileNameSuffix : String) {
        // Depth map is 32 bit float
        
        guard CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly) == noErr else { return }
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        guard let srcPtr = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            print("Failed to retrieve depth pointer.")
            return
        }
    
        let rowBytes : Int = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let width = Int(CVPixelBufferGetWidth(pixelBuffer))
        let height = Int(CVPixelBufferGetHeight(pixelBuffer))
        let capacity = CVPixelBufferGetDataSize(pixelBuffer)
        let uint8Pointer = srcPtr.bindMemory(to: UInt8.self, capacity: capacity)
        
        let s = "depth_\(width)x\(height)_\(fileNameSuffix)"
        let fileURL = URL(fileURLWithPath: s, relativeTo: bundleFolder).appendingPathExtension("bin")
        
        guard let stream = OutputStream(url: fileURL, append: false) else {
            print("Failed to open depth stream.")
            return
        }
        stream.open()
        
        let header = "Time:\(String(describing: lastArData!.sampleTime!)),EulerAngles:\(lastArData!.eulerAngles.debugDescription)),WorldPose:\(String(describing: lastArData!.worldPose)),Intrinsics:\(String(describing: lastArData?.cameraIntrinsics)),WorldToCamera:\(String(describing: lastArData?.worldToCamera))<ENDHEADER>"
        let encodedHeader = [UInt8](header.utf8)
        stream.write(encodedHeader, maxLength: 1024) // 1024 bits of header
        
        for y in 0 ..< height{
            stream.write(uint8Pointer + (y * rowBytes), maxLength: Int(rowBytes))
        }
        
        stream.close()
    }
    
    func writeConfidence(pixelBuffer: CVPixelBuffer, fileNameSuffix : String) {
        // Depth map is 32 bit float
        
        guard CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly) == noErr else { return }
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        guard let srcPtr = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            print("Failed to retrieve depth pointer.")
            return
        }
    
        let rowBytes : Int = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let width = Int(CVPixelBufferGetWidth(pixelBuffer))
        let height = Int(CVPixelBufferGetHeight(pixelBuffer))
        let capacity = CVPixelBufferGetDataSize(pixelBuffer)
        let uint8Pointer = srcPtr.bindMemory(to: UInt8.self, capacity: capacity)
        
        let s = "conf_\(width)x\(height)_\(fileNameSuffix)"
        let fileURL = URL(fileURLWithPath: s, relativeTo: bundleFolder).appendingPathExtension("bin")
        
        guard let stream = OutputStream(url: fileURL, append: false) else {
            print("Failed to open depth stream.")
            return
        }
        stream.open()
        
        for y in 0 ..< height{
            stream.write(uint8Pointer + (y * rowBytes), maxLength: Int(rowBytes))
        }
        
        stream.close()
    }
    
    func writeHeaderOnly(fileNameSuffix : String) {
        // Write only camera header info
        
        let s = "info_\(fileNameSuffix)"
        let fileURL = URL(fileURLWithPath: s, relativeTo: bundleFolder).appendingPathExtension("bin")
        
        guard let stream = OutputStream(url: fileURL, append: false) else {
            print("Failed to open depth stream.")
            return
        }
        stream.open()
        
        let header = "Time:\(String(describing: lastArData!.sampleTime!)),EulerAngles:\(lastArData!.eulerAngles.debugDescription)),WorldPose:\(String(describing: lastArData!.worldPose)),Intrinsics:\(String(describing: lastArData?.cameraIntrinsics)),WorldToCamera:\(String(describing: lastArData?.worldToCamera))<ENDHEADER>"
        let encodedHeader = [UInt8](header.utf8)
        stream.write(encodedHeader, maxLength: 1024) // 1024 bits of header
        stream.close()
    }
    
    func initBundleFolder(suffix: String = "") {
        let currDate = Date()
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let currDateString = dateFormatter.string(from : currDate)
        
        let DocumentDirectory = URL(fileURLWithPath: NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)[0])
        let DirPath = DocumentDirectory.appendingPathComponent("bundle-" + currDateString + suffix + "/")
        
        do {
            try FileManager.default.createDirectory(atPath: DirPath.path, withIntermediateDirectories: true, attributes: nil)
        } catch let error as NSError {
            print("Unable to create directory \(error.debugDescription)")
        }
        
        bundleFolder = URL(fileURLWithPath: DirPath.path)
    }
    
    func recordPoseBundle(saveSuffix: String = ""){
        recordScene = false
        frameCount = -20 // skip first 20 frames
        if saveSuffix != "" {
            initBundleFolder(suffix: "-" + saveSuffix + "-poses")
        } else {
            initBundleFolder(suffix: "-poses")
        }
        print("Recording poses into \(String(describing: bundleFolder!.path))")
    }
    
    func recordBundle(saveSuffix: String = ""){
        recordScene = true
        frameCount = -20 // skip first 20 frames
        if saveSuffix != "" {
            initBundleFolder(suffix: "-" + saveSuffix)
        } else {
            initBundleFolder()
        }
        print("Recording bundle into \(String(describing: bundleFolder!.path))")
    }
    
    // arData access point
    // Save a reference to the current AR data and process it.
    var ind = 0
    func onNewARData(arData: ARData) {
        lastArData = arData
        processLastArData()
    }
    
    // Copy the AR data to Metal textures and, if the user enables the UI, upscale the depth using a guided filter.
    func processLastArData() {
        if frameCount < bundleSize {
            if frameCount >= 0 {
                if recordScene {
                    writeDepth(pixelBuffer : lastArData!.depthImage!, fileNameSuffix : "\(frameCount)")
                    writeImageYUV(pixelBuffer : lastArData!.colorImage!, fileNameSuffix : "\(frameCount)")
                    writeConfidence(pixelBuffer: lastArData!.confidenceImage!, fileNameSuffix: "\(frameCount)")
                } else {
                    writeHeaderOnly(fileNameSuffix: "\(frameCount)")
                }
            }
            frameCount += 1
        }
        else {
            frameCount = 99999
        }
        
        
        
        colorYContent.texture = lastArData?.colorImage?.texture(withFormat: .r8Unorm, planeIndex: 0, addToCache: textureCache!)!
        colorCbCrContent.texture = lastArData?.colorImage?.texture(withFormat: .rg8Unorm, planeIndex: 1, addToCache: textureCache!)!
        depthContent.texture = lastArData?.depthImage?.texture(withFormat: .r32Float, planeIndex: 0, addToCache: textureCache!)!


        guard let commandQueue = commandQueue else { return }
        guard let cmdBuffer = commandQueue.makeCommandBuffer() else { return }
        guard let computeEncoder = cmdBuffer.makeComputeCommandEncoder() else { return }
        // Convert YUV to RGB because the guided filter needs RGB format.
        computeEncoder.setComputePipelineState(pipelineStateCompute!)
        computeEncoder.setTexture(colorYContent.texture, index: 0)
        computeEncoder.setTexture(colorCbCrContent.texture, index: 1)
        computeEncoder.setTexture(colorRGBTexture, index: 2)
        let threadgroupSize = MTLSizeMake(pipelineStateCompute!.threadExecutionWidth,
                                          pipelineStateCompute!.maxTotalThreadsPerThreadgroup / pipelineStateCompute!.threadExecutionWidth, 1)
        let threadgroupCount = MTLSize(width: Int(ceil(Float(colorRGBTexture.width) / Float(threadgroupSize.width))),
                                       height: Int(ceil(Float(colorRGBTexture.height) / Float(threadgroupSize.height))),
                                       depth: 1)
        computeEncoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        computeEncoder.endEncoding()
        cmdBuffer.commit()
        
        colorRGBContent.texture = colorRGBTexture
        
    }
}

