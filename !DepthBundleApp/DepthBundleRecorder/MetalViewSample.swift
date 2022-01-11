/*
 See LICENSE folder for this sample’s licensing information.
 
 Abstract:
 A parent view class that displays the sample app's other views.
 */

import Foundation
import SwiftUI
import MetalKit
import ARKit

// Add a title to a view that enlarges the view to full screen on tap.
struct Texture<T: View>: ViewModifier {
    let height: CGFloat
    let width: CGFloat
    let title: String
    let view: T
    func body(content: Content) -> some View {
        VStack {
            Text(title).foregroundColor(Color.red)
            // To display the same view in the navigation, reference the view
            // directly versus using the view's `content` property.
            NavigationLink(destination: view.aspectRatio(CGSize(width: width, height: height), contentMode: .fill)) {
                view.frame(maxWidth: width, maxHeight: height, alignment: .center)
                    .aspectRatio(CGSize(width: width, height: height), contentMode: .fill)
            }
        }
    }
}

extension View {
    // Apply `zoomOnTapModifier` with a `self` reference to show the same view
    // on tap.
    func zoomOnTapModifier(height: CGFloat, width: CGFloat, title: String) -> some View {
        modifier(Texture(height: height, width: width, title: title, view: self))
    }
}
extension Image {
    init(_ texture: MTLTexture, ciContext: CIContext, scale: CGFloat, orientation: Image.Orientation, label: Text) {
        let ciimage = CIImage(mtlTexture: texture)!
        let cgimage = ciContext.createCGImage(ciimage, from: ciimage.extent)
        self.init(cgimage!, scale: 1.0, orientation: .leftMirrored, label: label)
    }
}
//- Tag: MetalDepthView
struct MetalDepthView: View {
    
    // Set the default sizes for the texture views.
    let sizeH: CGFloat = 256
    let sizeW: CGFloat = 192
    
    // Manage the AR session and AR data processing.
    //- Tag: ARProvider
    @ObservedObject var arProvider: ARProvider = ARProvider()!
    let ciContext: CIContext = CIContext()
    
    // Save the user's confidence selection.
    @State private var selectedConfidence = 0
    // Set the depth view's state data.
    @State var isToUpsampleDepth = false
    @State var isShowSmoothDepth = false
    @State var isArPaused = false
    @State private var scaleMovement: Float = 1.5
    @State var saveSuffix: String = ""
    @State var numRecordedSceneBundles = 0
    @State var numRecordedPoseBundles = 0
    
    var body: some View {
        if !ARWorldTrackingConfiguration.supportsFrameSemantics([.sceneDepth, .smoothedSceneDepth]) {
            Text("Unsupported Device: This app requires the LiDAR Scanner to access the scene's depth.")
        } else {
            VStack(alignment: .leading, spacing: 0) {
                
                // bundle size selector
                HStack {
                    Spacer(minLength: 50)
                    Text("Bundle Size:")
                    Picker(selection: $arProvider.bundleSize, label: Text("Bundle Size:")) {
                        Text("15").tag(15)
                        Text("30").tag(30)
                        Text("60").tag(60)
                        Text("120").tag(120)
                        }.pickerStyle(SegmentedPickerStyle())
                }.frame(width: 350, height:50)
                
                // input field
                HStack() {
                Spacer(minLength: 40)
                Text("Recorded \(numRecordedPoseBundles) Pose, \(numRecordedSceneBundles) Scene Bundles")
                Spacer(minLength: 40)
                }.frame(width: 400, height: 40)
                
                // input field
                HStack() {
                Spacer(minLength: 60)
                TextField("Save File Suffix", text: $saveSuffix)
                        .disableAutocorrection(true)
                        .border(Color(UIColor.separator))
                        .autocapitalization(.none)
                Spacer(minLength: 60)
                }.frame(width: 400, height: 40)
                
                // depth and image display
                HStack(alignment: .top) {
                    Spacer(minLength: 50)
                    MetalTextureViewDepth(mtkView: MTKView(), content: arProvider.depthContent, confSelection: $selectedConfidence)
                    MetalTextureView(mtkView: MTKView(), content: arProvider.colorRGBContent)
                }.frame(width: 350, height:300)
                
                // buttons for stream interaction
                HStack() {
                    Spacer()
                    Button(action: {
                        isArPaused.toggle()
                        isArPaused ? arProvider.pause() : arProvider.start()
                    }) {
                        Image(systemName: isArPaused ? "play.circle" : "pause.circle").resizable().frame(width: 30, height: 30)
                    }
                    Spacer()
                    Button(action: {
                        if arProvider.frameCount == 99999 {
                            arProvider.recordPoseBundle(saveSuffix: saveSuffix)
                            numRecordedPoseBundles += 1
                        }
                    }) {
                        Image(systemName: (arProvider.frameCount == 99999) ? "move.3d" : "" )
                            .resizable().frame(width: 30, height: 30)
                    }
                    Spacer()
                    Button(action: {
                        if arProvider.frameCount == 99999 {
                            arProvider.recordBundle(saveSuffix: saveSuffix)
                            numRecordedSceneBundles += 1
                        }
                    }) {
                        Image(systemName: (arProvider.frameCount == 99999) ? "record.circle.fill" : "" )
                            .resizable().frame(width: 30, height: 30)
                    }
                    Spacer()
                }.frame(width: 400, height: 50)
            }.frame(maxWidth: .infinity, maxHeight: .infinity)
            .ignoresSafeArea()
        }
    }
}
//struct MtkView_Previews: PreviewProvider {
//    static var previews: some View {
//        Group {
//            MetalDepthView().previewDevice("iPhone 12 Max")
//        }
//    }
//}
